# Edinburgh Sample Download Attempt Log

## Target sample

- File: `pigs161119.zip`
- File ID: `1jZauXZ18-UiB9fxK70NB5lPSoG3v7e4U`
- Listed size: `2.6`
- Listed clips: `10`
- Clip index range: `67-76`

## Attempt result

The Google Drive folder listing was successfully retrieved with `gdown --folder --json`, but direct file download failed because Google Drive returned a quota / too many users message.

## Interpretation

The file link is valid, but automated server-side download is temporarily restricted by Google Drive. This does not invalidate the dataset. For Week 4 dataset exploration, the source page metadata and file manifest can still be used. If sample-level inspection is needed, the sample zip can be downloaded manually through a browser and transferred to the server.

## Next action

Manually download `pigs161119.zip` from the browser if accessible, then upload it to:

`~/PigBench/Week4_5_Behaviour_Representation/data/public_datasets/edinburgh/samples/pigs161119.zip`
