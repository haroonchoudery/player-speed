import csv
import os


DIR = "/Users/haroonchoudery/player-speed/hart-code/Annotate/data/batch_0"

max_height = 0
max_width = 0

for filename in os.listdir(DIR):
    if filename.endswith(".txt"):
        with open(os.path.join(DIR, filename), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for index, row in enumerate(reader):
                if index == 0:
                    FT_LEFT_x = float(row[1])
                    FT_LEFT_y = float(row[2])
                elif index == 1:
                    FT_RIGHT_x = float(row[1])
                    FT_RIGHT_y = float(row[2])
                elif index == 2:
                    BL_RIGHT_x = float(row[1])
                    BL_RIGHT_y = float(row[2])
                else:
                    BL_LEFT_x = float(row[1])
                    BL_LEFT_y = float(row[2])

    if abs(FT_LEFT_x - BL_LEFT_x) > max_width:
        max_width = abs(FT_LEFT_x - BL_LEFT_x)

    if abs(FT_RIGHT_x - BL_RIGHT_x) > max_width:
        max_width = abs(FT_RIGHT_x - BL_RIGHT_x)

    if abs(FT_LEFT_y - FT_RIGHT_y) > max_height:
        max_height = abs(FT_LEFT_y - FT_RIGHT_y)


print("Max Width:", max_width * 1024)
print("Max Height:", max_height * 576)
