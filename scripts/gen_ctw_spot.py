import json
from pathlib import Path
base = Path("/home/yzy/mmocr/data")
for split in ["train","test"]:
    det_file = base / f"ctw1500_textdet_{split}.json"
    out_file = base / f"ctw1500_textspotting_{split}.json"
    if not det_file.exists():
        print("MISSING", det_file)
        continue
    with open(det_file, r, encoding=utf-8) as f:
        data = json.load(f)
    dl = data.get(data_list, [])
    for item in dl:
        for inst in item.get(instances, []):
            if text not in inst:
                inst[text] = 
    meta = {
        metainfo: {
            dataset_type: TextSpottingDataset,
            task_name: textspotting,
            category: [{id: 0, name: text}]
        },
        data_list: dl
    }
    with open(out_file, w, encoding=utf-8) as f:
        json.dump(meta, f, ensure_ascii=False)
    print(WROTE, out_file, len(dl))
PY
python /home/yzy/mmocr/scripts/gen_ctw_spot.py | cat
