import carla

client = carla.Client("localhost", 2000)
client.set_timeout(5.0)
world = client.get_world()

# #건물 = Buildings, 정적 요소 = Static, 가로등 = Poles, 정적 객체 = Other, 나무 = Vegetation, 정적 차 = Vehicles


env_objs = world.get_environment_objects(carla.CityObjectLabel.Poles)


# # 모든 건물의 ID를 set으로 수집
building_ids = {obj.id for obj in env_objs}

# 모든 건물 숨기기
world.enable_environment_objects(building_ids, False)

print(f"{len(building_ids)}개의 ploe이 비활성화되었습니다.")

env_objs = world.get_environment_objects(carla.CityObjectLabel.Buildings)


# 모든 건물의 ID를 set으로 수집
building_ids = {obj.id for obj in env_objs}

# 모든 건물 숨기기
world.enable_environment_objects(building_ids, False)

print(f"{len(building_ids)}개의 나무이 비활성화되었습니다.")



env_objs = world.get_environment_objects(carla.CityObjectLabel.TrafficLight)


# 모든 건물의 ID를 set으로 수집
building_ids = {obj.id for obj in env_objs}

# 모든 건물 숨기기
world.enable_environment_objects(building_ids, False)

print(f"{len(building_ids)}개의 나무이 비활성화되었습니다.")
