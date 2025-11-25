from roboflow import Roboflow

rf = Roboflow(api_key="hxyw1unCnB8CGuhiNAxV")
project = rf.workspace("Maxis").project("PROJECT_NAME")
dataset = project.version(VERSION_NUMBER).download("FORMAT")