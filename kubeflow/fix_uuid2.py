import uuid, re
new_id = str(uuid.uuid4())
with open('wf_fresh.yaml') as f:
    content = f.read()
# remove entire broken string including trailing junk
content = re.sub(r'\$\(python3[^"]*\)"?\)?', new_id, content)
with open('wf_fresh.yaml', 'w') as f:
    f.write(content)
print('done, new id:', new_id)
