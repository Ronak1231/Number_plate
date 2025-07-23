yaml_content = """
train: indian_number_plates/images/train
val: indian_number_plates/images/val

nc: 1
names: ['number_plate']
"""

with open("data.yaml", "w") as f:
    f.write(yaml_content.strip())

print("✅ data.yaml created")
