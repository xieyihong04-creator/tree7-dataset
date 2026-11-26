import csv, io, collections
import pyarrow.parquet as pq
from PIL import Image

man = collections.defaultdict(list)
with open('/tmp/manifest.csv', newline='', encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        k = (int(r['class_index']), int(r['bytes']), int(r['w']), int(r['h']))
        man[k].append(r)

pf = pq.ParquetFile('/tmp/tree7/test.parquet')
print('rows', pf.metadata.num_rows)
hit = miss = amb = 0
examples = []
for batch in pf.iter_batches(batch_size=64, columns=['image', 'label']):
    imgs = batch.column(0).to_pylist()
    labels = batch.column(1).to_pylist()
    for img, label in zip(imgs, labels):
        data = img['bytes'] if isinstance(img, dict) else img.as_py()['bytes']
        im = Image.open(io.BytesIO(data))
        k = (int(label), len(data), im.width, im.height)
        cands = man.get(k, [])
        if len(cands) == 1:
            hit += 1
        elif len(cands) == 0:
            miss += 1
            if len(examples) < 5:
                examples.append((k, 'miss'))
        else:
            amb += 1
            if len(examples) < 5:
                examples.append((k, f'amb {len(cands)}'))
print({'unique_hit': hit, 'miss': miss, 'ambiguous': amb})
print('examples', examples)
