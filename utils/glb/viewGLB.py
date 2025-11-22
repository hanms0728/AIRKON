import trimesh

filename = "utils/glb/street_lamp_hanging.glb"

obj = trimesh.load(filename)

print(type(obj))

# Trimesh인 경우 → 바로 scene() 사용
if isinstance(obj, trimesh.Trimesh):
    scene = obj.scene()
    scene.show()

# Scene인 경우 → 바로 show() 사용 (여기서 에러 안 남)
elif isinstance(obj, trimesh.Scene):
    obj.show()

else:
    # list나 다른 타입이면 안전하게 씬으로 합치기
    merged = trimesh.util.concatenate(obj)
    merged.scene().show()
