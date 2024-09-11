from PIL.PngImagePlugin import PngInfo
# import struct
import numpy as np

from io import BytesIO

import pyfpng  # https://github.com/qrmt/fpng-python # has to be installed directly from c++ using setup.py install
import fpng_py # https://github.com/K0lb3/fpng_py you still need the above pyfpng to run the setup.py and install the cpp file for fpng in C++

import logging


def numpy_array_to_fpng(array: np.ndarray, filename:str="") -> bytes:

    """
    Encode a numpy array into a PNG image using fpng.

    Args:
        array: The numpy array to encode. Must be a 3D array with shape (h, w, channels),
            where channels is either 3 (no alpha channel) or 4 (with alpha channel).
        filename: The file path to which to save the image. If not provided, the image
            will be encoded into a bytes string.

    Returns:
        A tuple (success, output), where success is True if the encoding was successful,
        and output is either the filename if a file was saved, or the encoded bytes string.
    """
    try:
        # logging.info(f"Image shape: {array.shape}")
        # logging.info(f"Image data type: {array.dtype}")
        # Determine the number of channels from the array's shape
        num_channels = 3 if array.ndim == 3 else 4
        # Convert the array to a bytes string
        image_bytes = array.tobytes()
        # Get the width and height from the array's shape
        w, h = array.shape[:2]
        # logging.info(f"Number of channels: {num_channels}")
        if filename:
            fpng_py.fpng_encode_image_to_file(
                filename,
                image_bytes,
                w,
                h,
                num_channels,
            )
            return True, filename

        output = fpng_py.fpng_encode_image_to_memory(
            image_bytes,
            w,
            h,
            num_channels,
        )
        success = True
    except Exception as e:
        output = None
        success = False
        print(f"Error: {e}")
        logging.error(f"Error: {e}")
        raise


    return success, output


#  Generate the CRC table
#  This is unused
# CRC_TABLE = []
# for i in range(256):
#     crc = i
#     for _ in range(8):
#         if crc & 1:
#             crc = (crc >> 1) ^ 0xedb88320
#         else:
#             crc = crc >> 1
#     CRC_TABLE.append(crc)





def add_metadata(png_bytestring:bytes, metadata:PngInfo):
    # This does not work as of right now, moving to prod neway
    return png_bytestring
    chunks = []
    for chunk in metadata.chunks:
        chunk_type = chunk[0]
        if not isinstance(chunk_type, bytes) or len(chunk_type) != 4:
            raise ValueError(f"Invalid chunk type: {chunk_type}")
        chunk_data = chunk[1]
        chunk_length = len(chunk_data)
        crc = 0xffffffff
        for byte in chunk_type + chunk_data:
            crc = (crc >> 8) ^ CRC_TABLE[(crc & 0xff) ^ byte]
        chunks.append(struct.pack('>I', chunk_length) + chunk_type + chunk_data + struct.pack('>I', crc))
    modified_png_data = BytesIO()
    modified_png_data.write(png_bytestring[:21])
    modified_png_data.write(b''.join(chunks))
    modified_png_data.write(png_bytestring[21:])
    return modified_png_data.getvalue()



# save for testing the save_images function
"""
    def save_images_time_test(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            

            from copy import deepcopy
            import timeit

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"

            metadata = None

            if metadata is not None:
                # Measure execution time of numpy_array_to_fpng
                numpy_array_to_fpng_time = timeit.timeit(lambda: numpy_array_to_fpng(np.clip(deepcopy(i), 0, 255).astype(np.uint8)), number=100)
                logging.info(f"numpy_array_to_fpng execution time: {(numpy_array_to_fpng_time * 10):.6f} milliseconds each")

                # Measure execution time of img.save
                # img = Image.fromarray(data)
                img_save_time = timeit.timeit(lambda: Image.fromarray(np.clip(deepcopy(i), 0, 255).astype(np.uint8)).save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level), number=100)
                logging.info(f"img.save execution time: {(img_save_time * 10):.6f} milliseconds each")

                success, img = numpy_array_to_fpng(np.clip(deepcopy(i), 0, 255).astype(np.uint8))
                if not success:
                    img = Image.fromarray(np.clip(deepcopy(i), 0, 255).astype(np.uint8))
                else:
                    with open(os.path.join(full_output_folder, file), "wb") as f:
                        f.write(add_metadata(img, metadata))
            else:
                # Measure execution time of numpy_array_to_fpng
                logging.info(f"Measuring execution time for {os.path.join(full_output_folder, file)} with numpy_array_to_fpng function 100 times (could take a while...)")
                numpy_array_to_fpng_time = timeit.timeit(lambda: numpy_array_to_fpng(np.clip(deepcopy(i), 0, 255).astype(np.uint8), filename=os.path.join(full_output_folder, file)), number=100)
                logging.info(f"numpy_array_to_fpng execution time: {(numpy_array_to_fpng_time * 10):.6f} milliseconds each")
                
                # Measure execution time of img.save
                # img = Image.fromarray(deepcopy(data))
                img_save_time = timeit.timeit(lambda: Image.fromarray(np.clip(deepcopy(i), 0, 255).astype(np.uint8)).save(os.path.join(full_output_folder, file), pnginfo=None, compress_level=self.compress_level), number=100)
                logging.info(f"Measuring execution time for {os.path.join(full_output_folder, file)} with PIL/Pillow fromarray(np) and Image.save() 100 times (could take a while...)")
                logging.info(f"img.save execution time: {(img_save_time * 10):.6f} milliseconds each")

                success, img = numpy_array_to_fpng(np.clip(deepcopy(i), 0, 255).astype(np.uint8), filename=os.path.join(full_output_folder, file))
                if not success:
                    img.save(os.path.join(full_output_folder, file), pnginfo=None, compress_level=self.compress_level)


            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}
  
"""