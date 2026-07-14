# Edinburgh `output.json` Schema Inspection

## Purpose

This step inspects one manually ground-truthed Edinburgh `output.json` file directly from `annotated.tar`, without extracting the full archive.

## Sample inspected

- `annotated/2019_11_15/000033/output.json`

## Schema summary

| field                   | value                                                       |
|:------------------------|:------------------------------------------------------------|
| sample_member           | annotated/2019_11_15/000033/output.json                     |
| top_level_type          | dict                                                        |
| top_level_keys          | videoFileName, fullVideoFilePath, stepSize, config, objects |
| dict_count_recursive    | 1968                                                        |
| list_count_recursive    | 9                                                           |
| bbox_count              | 979                                                         |
| unique_behaviour_labels | 8                                                           |
| min_frameNumber         | 0                                                           |
| max_frameNumber         | 599                                                         |
| frameNumber_count       | 979                                                         |

## Behaviour label counts

| behaviour     |   count |
|:--------------|--------:|
| standing      |     438 |
| walk          |     339 |
| investigating |      85 |
| sleep         |      63 |
| sitting       |      24 |
| drink         |      18 |
| playwithtoy   |       8 |
| lying         |       4 |

## Most common JSON keys

| key               |   count |
|:------------------|--------:|
| frameNumber       |     979 |
| bbox              |     979 |
| isGroundTruth     |     979 |
| visible           |     979 |
| behaviour         |     979 |
| x                 |     979 |
| y                 |     979 |
| width             |     979 |
| height            |     979 |
| frames            |       8 |
| id                |       8 |
| stepSize          |       2 |
| videoFileName     |       1 |
| fullVideoFilePath |       1 |
| config            |       1 |
| objects           |       1 |
| playbackRate      |       1 |
| imageMimeType     |       1 |
| imageExtension    |       1 |
| framesZipFilename |       1 |
| consoleLog        |       1 |

## Largest lists in JSON

| path                   |   length |
|:-----------------------|---------:|
| root.objects[4].frames |      156 |
| root.objects[5].frames |      139 |
| root.objects[2].frames |      128 |
| root.objects[6].frames |      128 |
| root.objects[0].frames |      114 |
| root.objects[3].frames |      113 |
| root.objects[1].frames |      103 |
| root.objects[7].frames |       98 |
| root.objects           |        8 |

## Bounding box summary

|   bbox_count |   mean_width |   mean_height |   min_x |   max_x |   min_y |   max_y |
|-------------:|-------------:|--------------:|--------:|--------:|--------:|--------:|
|          979 |      245.544 |       224.544 |     140 |    1090 |       0 |     511 |

## Interpretation

This inspection confirms the internal structure of the Edinburgh manually annotated ground-truth file. The extracted behaviour labels, bounding box fields, frame numbers, visibility flags, and ground-truth flags can be compared directly with our Unibo Week 3 corrected annotation JSON.
