class PoseBonesInspector(object):
    def __init__(self, renv):
        self.renv = renv
 
    def inspect(self):
        return False


def do_inspect(renv):
    print("##inspect_posebones")

    ski = PoseBonesInspector(renv)
    return ski.inspect()
