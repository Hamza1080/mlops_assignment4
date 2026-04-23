import uuid
new_id = str(uuid.uuid4())
with open('wf_fresh.yaml') as f:
    content = f.read()
# fix the broken substitution in labels
import re
content = re.sub(r'\$\(python3[^)]+\)\)', new_id, content)
content = content.replace('fraud-detection-pipeline-fresh10', 'fraud-detection-pipeline-fresh11')
with open('wf_fresh.yaml', 'w') as f:
    f.write(content)
print('new id:', new_id)
