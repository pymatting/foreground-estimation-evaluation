import struct
import numpy as np


def packbits_decode(src, dst):
    i = 0
    j = 0
    while i < len(src):
        n = src[i]
        n -= (n >> 7) << 8
        i += 1

        if n == -128:
            continue

        elif n >= 0:
            n += 1

            dst[j : j + n] = src[i : i + n]
            i += n
            j += n
        else:
            n = 1 - n

            dst[j : j + n] = src[i : i + 1] * n
            i += 1
            j += n


def read_tiff(path):
    # Only supports a very specific subset of TIFF images.
    # For TIFF file format, see TIFF Revision 6.0
    # https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.pdf

    tag_names = {
        0x100: "ImageWidth",
        0x101: "ImageLength",
        0x102: "BitsPerSample",
        0x103: "Compression",
        0x106: "PhotometricInterpretation",
        0x111: "StripOffsets",
        0x115: "SamplesPerPixel",
        0x116: "RowsPerStrip",
        0x117: "StripByteCounts",
        0x11A: "XResolution",
        0x11B: "YResolution",
        0x128: "ResolutionUnit",
        0x131: "Software",
        0x132: "DateTime",
    }

    field_type_to_data_type = {
        1: "B",
        2: "B",
        3: "H",
        4: "I",
        5: "II",
    }

    field_type_byte_size = {
        1: 1,
        2: 1,
        3: 2,
        4: 4,
        5: 8,
    }

    with open(path, "rb") as f:
        # Only little endian supported
        assert f.read(2) == b"II"
        # Magic byte to indicate that this is a TIFF file
        assert f.read(2) == b"\x2a\x00"
        # Offset where IFD (image file directory) starts
        offset = struct.unpack("<I", f.read(4))[0]
        f.seek(offset)

        # Read tags
        number_of_tags = struct.unpack("<H", f.read(2))[0]
        tags = {}
        for _ in range(number_of_tags):
            tag = f.read(12)

            tag_id, field_type, count = struct.unpack("<HHI", tag[:8])

            size = count * field_type_byte_size[field_type]

            # Last 4 bytes of tag store the tag value(s)
            if size <= 4:
                values = tag[8 : 8 + size]
            else:
                # If tag values are more than 4 bytes, then the last 4 bytes
                # of the tag store the offset where the values are stored.
                (offset,) = struct.unpack("<I", tag[-4:])

                offset_backup = f.tell()
                f.seek(offset)
                values = f.read(size)
                f.seek(offset_backup)

            # Convert tag values
            value_format = "<" + field_type_to_data_type[field_type] * count
            tag_name = tag_names.get(tag_id, "Unknown tag")
            tags[tag_name] = struct.unpack(value_format, values)
            # print("%25s" % tag_name, hex(tag_id), field_type, count, values[:10])

        # Only a single IFD is supported, so the next IDF offset must be 0
        next_offset = struct.unpack("<I", f.read(4))[0]
        assert next_offset == 0

        width = tags["ImageWidth"][0]
        height = tags["ImageLength"][0]
        rows_per_strip = tags["RowsPerStrip"][0]
        if tags["Compression"][0] not in [1, 32773]:
            raise NotImplementedError("TIFF compression format not supported")

        strips = []
        remaining_rows = height
        for offset, length in zip(tags["StripOffsets"], tags["StripByteCounts"]):
            f.seek(offset)

            strip = f.read(length)

            if tags["Compression"][0] == 32773:
                tmp = bytearray(min(rows_per_strip, remaining_rows) * width * 6)
                packbits_decode(strip, tmp)
                strip = bytes(tmp)
                remaining_rows -= rows_per_strip

            strips.append(strip)

        image = np.frombuffer(b"".join(strips), dtype=np.uint16)

        return image.reshape(height, width, 3)


if __name__ == "__main__":
    import os

    directory = os.path.dirname(os.path.abspath(__file__))

    for path in [
        os.path.join(directory, "test_17x11_packbits_16bit.tif"),
        os.path.join(directory, "test_17x11_no_compression_16_bit.tif"),
    ]:
        image = read_tiff(path)

        import matplotlib.pyplot as plt

        image = image / 65535.0
        plt.imshow(image)
        plt.show()
