# 轨迹说明

## raw_examples

这里保留完整原始 JSON 行，包含 prompt、proposal、merge、relation judge、local debate、terminality debate、final path debate 等字段。适合逐字段审阅。

## compact

这里保留压缩版本，只抽取问题、答案、树节点、层摘要、终点辩论、三投影选择等关键字段。适合快速看算法行为。

## 样例清单

| 文件 | 版本 | 用途 |
| --- | --- | --- |
| `v0_initial_gold_context_pilot.jsonl` | v0 | 最初 LDT 小样本完整轨迹 |
| `v8_q3_8b_musique_success_fletcher.json` | v8 | terminality / path debate 成功样例 |
| `v8_q3_8b_musique_success_knowsley.json` | v8 | town 到 borough 的终点粒度成功样例 |
| `v8_q3_8b_musique_failure_adelphia_successor.json` | v8 | 描述性占位答案失败样例 |
| `v23_q25_14b_2wiki_success_rupert_full500.json` | v23 | father-in-law 角色方向成功样例 |
| `v23_q25_14b_musique_failure_whiston_full50.json` | v23 | guard 后仍失败的粒度样例 |
| `v23_q25_14b_full500_first_failures_compact.jsonl` | v23 | 最新 full500 首批失败压缩摘要 |
| `v62_q35_musique_success_fletcher_full500.json` | v6.2 | 三投影 evidence plan 成功样例 |
| `v62_q35_musique_success_knowsley_full500.json` | v6.2 | 三投影终端选择成功样例 |

注意：raw trace 体积会较大。正式上传前可以只保留 compact 版本和少量 raw 版本。

