# Aggressive Pig/Chicken Dataset Access Audit

## Purpose

This note strengthens the public dataset exploration section by documenting what could and could not be inspected from the aggressive pig/chicken behaviour dataset repository. The goal is to be explicit about dataset accessibility, annotation visibility, and why this dataset is treated as a video-temporal recognition reference rather than a trajectory/ROI feature dataset.

## Repository file inventory

| relative_path                                                        |   size_bytes | suffix   |
|:---------------------------------------------------------------------|-------------:|:---------|
| .git/HEAD                                                            |           21 |          |
| .git/config                                                          |          289 |          |
| .git/description                                                     |           73 |          |
| .git/hooks/applypatch-msg.sample                                     |          478 | .sample  |
| .git/hooks/commit-msg.sample                                         |          896 | .sample  |
| .git/hooks/fsmonitor-watchman.sample                                 |         4726 | .sample  |
| .git/hooks/post-update.sample                                        |          189 | .sample  |
| .git/hooks/pre-applypatch.sample                                     |          424 | .sample  |
| .git/hooks/pre-commit.sample                                         |         1643 | .sample  |
| .git/hooks/pre-merge-commit.sample                                   |          416 | .sample  |
| .git/hooks/pre-push.sample                                           |         1374 | .sample  |
| .git/hooks/pre-rebase.sample                                         |         4898 | .sample  |
| .git/hooks/pre-receive.sample                                        |          544 | .sample  |
| .git/hooks/prepare-commit-msg.sample                                 |         1492 | .sample  |
| .git/hooks/push-to-checkout.sample                                   |         2783 | .sample  |
| .git/hooks/sendemail-validate.sample                                 |         2308 | .sample  |
| .git/hooks/update.sample                                             |         3650 | .sample  |
| .git/index                                                           |          209 |          |
| .git/info/exclude                                                    |          240 |          |
| .git/logs/HEAD                                                       |          217 |          |
| .git/logs/refs/heads/main                                            |          217 |          |
| .git/logs/refs/remotes/origin/HEAD                                   |          217 |          |
| .git/objects/pack/pack-170bb49abb5bca177701ab0afb09640aa082d1c9.idx  |         2108 | .idx     |
| .git/objects/pack/pack-170bb49abb5bca177701ab0afb09640aa082d1c9.pack |        14464 | .pack    |
| .git/objects/pack/pack-170bb49abb5bca177701ab0afb09640aa082d1c9.rev  |          200 | .rev     |
| .git/packed-refs                                                     |          112 |          |
| .git/refs/heads/main                                                 |           41 |          |
| .git/refs/remotes/origin/HEAD                                        |           30 |          |
| LICENSE                                                              |        11357 |          |
| README.md                                                            |         1032 | .md      |

## Extracted README links

| extracted_url                                   | contains_baidu   | interpretation                 |
|:------------------------------------------------|:-----------------|:-------------------------------|
| https://pan.baidu.com/s/1Ez24CfSI8CdZ3Ze3046N0g | True             | External dataset/download link |
| https://pan.baidu.com/s/1yVvppXdGHHBX9b0td4J0mQ | True             | External dataset/download link |
| https://pan.baidu.com/s/1-sccboe7GGFGJ21xZOCifA | True             | External dataset/download link |

## Baidu / external access evidence

- link：https://pan.baidu.com/s/1Ez24CfSI8CdZ3Ze3046N0g
- 链接: https://pan.baidu.com/s/1yVvppXdGHHBX9b0td4J0mQ 提取码: 1234
- 链接: https://pan.baidu.com/s/1-sccboe7GGFGJ21xZOCifA 提取码: 1234


## Audit table

| criterion                                              | status      | evidence                                                                                                                                       |
|:-------------------------------------------------------|:------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
| Repository accessible locally                          | yes         | Week4_5_Behaviour_Representation/data/public_datasets/aggressive_pig_chicken/pig-and-chicken-behavior-dataset                                  |
| README available                                       | yes         | Week4_5_Behaviour_Representation/data/public_datasets/aggressive_pig_chicken/pig-and-chicken-behavior-dataset/README.md                        |
| Actual dataset files visible in GitHub clone           | partial     | 30 files found in repository clone                                                                                                             |
| External Baidu links mentioned                         | yes         | Baidu-related lines or URLs found in README                                                                                                    |
| Annotation format directly inspectable                 | no          | No annotation files are visible in the cloned repository                                                                                       |
| Video clip duration directly inspectable               | no          | No video files are visible in the cloned repository                                                                                            |
| Behaviour classes directly inspectable from repository | partial     | README and TSM article indicate aggressive/non-aggressive style usage, but full class files are not visible locally                            |
| Suitability for trajectory/ROI feature engineering     | low         | Dataset appears more suitable for video-based behaviour recognition than trajectory-based representation because no tracking files are visible |
| Suitability for video-temporal representation review   | medium-high | Useful as a TSM/aggression-recognition reference dataset                                                                                       |

## Interpretation

The cloned GitHub repository does not provide directly inspectable video files, annotation files, or tracking outputs. Therefore, this dataset should not be used as the main source for trajectory or ROI feature engineering. It is more appropriate as a video-based behaviour recognition reference, especially for aggressive/non-aggressive temporal modelling. The main internal feature-engineering dataset remains the Unibo corrected scan-window dataset, while Edinburgh remains the stronger public reference for annotation structure and tracking-related behaviour representation.
