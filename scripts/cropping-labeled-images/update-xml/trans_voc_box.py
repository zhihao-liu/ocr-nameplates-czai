from pascal_voc_io import *
import os

xml_folder = "/Users/liuzhihao/Dropbox/Advancement/Internship Cuizhou/OCR/Labeling AlfaRomeo/full-size/xml-chars"
info_folder = "/Users/liuzhihao/Desktop/cropped/info"
updated_xml_folder = "/Users/liuzhihao/Desktop/cropped/xml"

for xml_name in os.listdir(xml_folder):
    xml_reader = PascalVocReader(os.path.join(xml_folder, xml_name))
    for i in [0, 1]:
        updated_file_name = xml_name[:-4] + '-cropped-' + str(i)

        with open(os.path.join(info_folder, updated_file_name + ".txt")) as f:
            roi = f.readlines()

        roi_left = int(roi[0])
        roi_top = int(roi[1])
        roi_right = int(roi[2])
        roi_bottom = int(roi[3])

        xml_reader.width = roi_right - roi_left
        xml_reader.height = roi_bottom - roi_top
        xml_writer = PascalVocWriter(updated_xml_folder, updated_file_name, [xml_reader.height, xml_reader.width])

        for shape in xml_reader.getShapes():
            label = shape[0]
            points = shape[1]
            rect_left = points[0][0]
            rect_top = points[0][1]
            rect_right = points[2][0]
            rect_bottom = points[2][1]

            if rect_left < roi_left or rect_top < roi_top or rect_right > roi_right or rect_bottom > roi_bottom:
                continue

            xml_writer.addBndBox(
                rect_left - roi_left,
                rect_top - roi_top,
                rect_right - roi_left,
                rect_bottom - roi_top,
                str(label)
            )

        xml_writer.save(updated_xml_folder + "/" + updated_file_name + XML_EXT)

print('DONE')