import os
import json 


DEFAULT_CFG = {
    "version": "0.0.1",
    "output_folder": "_gen_auto_mcd_beard",
}

class AutoRenderEnv():
    ENV_AR_MAX_VARIATION = "ENV_AR_MAX_VARIATION"
    ENV_AR_COMBINATION_ID = "ENV_AR_COMBINATION_ID"

class AutoRenderCfg():
    CFG_OUTPUT_FOLDER = "output_folder"

    @classmethod
    def _get_config_filename(cls, cfg_json_path):
        dir_this = os.path.dirname(__file__)
        json_full_path = "{}{}{}".format(dir_this, os.sep, cfg_json_path)
        if os.path.isfile(json_full_path):
            return json_full_path
        return None

    @classmethod
    def get_config_data(cls, cfg_json_path):
        json_full_path = cls._get_config_filename(cfg_json_path)
        if json_full_path is not None:
            with open(json_full_path, "rt") as f:
                json_data = json.load(f)
                return json_data
        return None

    @classmethod
    def _set_config_data(cls, cfg_json_path, dict_data):
        json_full_path = cls._get_config_filename(cfg_json_path)
        if json_full_path is not None:
            with open(json_full_path, "wt") as f:
                json.dump(dict_data, f, indent=4)
                return True
        return False

    @classmethod
    def update_config(cls, cfg_json_path, key_name, value):
        json_data = cls.get_config_data(cfg_json_path)
        if json_data is None:
            json_data = DEFAULT_CFG.copy()
        json_data[key_name] = value
        return cls._set_config_data(cfg_json_path, json_data)

    @classmethod
    def get_config_value(cls, cfg_json_path, key_name):
        json_data = cls.get_config_data(cfg_json_path)
        if json_data is None:
            return None
        if key_name in json_data:
            return json_data[key_name]
        return None


if __name__ == '__main__':
    from pprint import pprint

    cfg_json_path = "bldmc_render_m00/auto_render.json"

    name = AutoRenderCfg._get_config_filename(cfg_json_path)
    print(name)

    json_data = AutoRenderCfg.get_config_data(cfg_json_path)
    pprint(json_data)    

    ret = AutoRenderCfg._set_config_data(cfg_json_path, json_data)
    print(ret)

    ret = AutoRenderCfg.update_config(cfg_json_path, "test", False)
    print(ret)

    ret = AutoRenderCfg.get_config_value(cfg_json_path, "test")
    print(ret)
