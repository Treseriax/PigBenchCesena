# Edinburgh `pigs161119.zip` Sample Inspection

## Sample file

- Zip path: `Week4_5_Behaviour_Representation/data/public_datasets/edinburgh/samples/pigs161119.zip`
- Total entries: `101`
- File entries: `90`
- Directory entries: `11`

## Clip summary

|   clip_id |   file_count |   total_size_bytes |   has_color_mp4 |   has_depth_mp4 |   has_background_png |   has_background_depth_png |   has_mask_png |   has_times_txt |   has_output_json |
|----------:|-------------:|-------------------:|----------------:|----------------:|---------------------:|---------------------------:|---------------:|----------------:|------------------:|
|    000067 |            9 |          268409696 |               1 |               1 |                    1 |                          1 |              1 |               1 |                 0 |
|    000068 |            9 |          276561080 |               1 |               1 |                    1 |                          1 |              1 |               1 |                 0 |
|    000069 |            9 |          284674009 |               1 |               1 |                    1 |                          1 |              1 |               1 |                 0 |
|    000070 |            9 |          265926884 |               1 |               1 |                    1 |                          1 |              1 |               1 |                 0 |
|    000071 |            9 |          268224705 |               1 |               1 |                    1 |                          1 |              1 |               1 |                 0 |
|    000072 |            9 |          280808897 |               1 |               1 |                    1 |                          1 |              1 |               1 |                 0 |
|    000073 |            9 |          276783220 |               1 |               1 |                    1 |                          1 |              1 |               1 |                 0 |
|    000074 |            9 |          285433624 |               1 |               1 |                    1 |                          1 |              1 |               1 |                 0 |
|    000075 |            9 |          284392600 |               1 |               1 |                    1 |                          1 |              1 |               1 |                 0 |
|    000076 |            9 |          283299602 |               1 |               1 |                    1 |                          1 |              1 |               1 |                 0 |

## File type summary

| suffix   | file_name             |   count |   total_size_bytes |
|:---------|:----------------------|--------:|-------------------:|
| .mp4     | color.mp4             |      10 |         2604048664 |
| .mp4     | depth.mp4             |      10 |          152399215 |
| .npy     | depth_scale.npy       |      10 |               1360 |
| .npy     | inverse_intrinsic.npy |      10 |               2000 |
| .npy     | rot.npy               |      10 |               2000 |
| .png     | background.png        |      10 |           15899846 |
| .png     | background_depth.png  |      10 |            1627642 |
| .png     | mask.png              |      10 |              47590 |
| .txt     | times.txt             |      10 |             486000 |

## Annotation file check

`output.json` was not found inside this sample zip based on zip inventory. This suggests that this sample package contains raw source video/depth/background/mask files, while ground-truth or automatic result annotations may be distributed separately.
