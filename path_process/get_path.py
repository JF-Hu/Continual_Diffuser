import os


def get_project_path(project_name="continual_diffuser"):
    current_path = os.getcwd()
    paths = current_path.split('/')
    project_path = []
    for dic in paths:
        if dic != project_name:
            project_path.append(dic)
        else:
            project_path.append(dic)
            break
    return "/".join(project_path)


