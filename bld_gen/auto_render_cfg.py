import os
import json 

# {
#     "version": "0.0.1",
#     "output_folder": "_gen_auto_mcd_beard",
#     "max_variation": 4
# }


class AutoRenderCfg():
    cfg_json_path = "bldmc_render_mcd_beard/auto_render.json"

    @classmethod
    def get_config_filename(cls):
        dir_this = os.path.dirname(__file__)
        json_full_path = "{}{}{}".format(dir_this, os.sep, cls.cfg_json_path)
        if os.path.isfile(json_full_path):
            return json_full_path
        return None

    @classmethod
    def get_config_data(cls):
        json_full_path = cls.get_config_filename()
        if json_full_path is not None:
            with open(json_full_path, "rt") as f:
                json_data = json.load(f)
                return json_data
        return None

    @classmethod
    def set_config_data(cls, dict_data):
        json_full_path = cls.get_config_filename()
        if json_full_path is not None:
            with open(json_full_path, "wt") as f:
                json.dump(dict_data, f, indent=4)
                return True
        return False

if __name__ == '__main__':
    from pprint import pprint
    name = AutoRenderCfg.get_config_filename()
    print(name)

    json_data = AutoRenderCfg.get_config_data()
    pprint(json_data)    

    ret = AutoRenderCfg.set_config_data(json_data)
    print(ret)
