class WndFinder():
    @classmethod
    def inspect_wnd_region(cls):
        import bpy
        context = bpy.context
        for iw, window in enumerate(context.window_manager.windows):
            print(iw, window, window.parent, window.x, window.y, window.width, window.height)
            for ia, area in enumerate(window.screen.areas):
                print("\t",ia, area, area.type, area.x, area.y, area.width, area.height)
                for ir, region in enumerate(area.regions):
                    print("\t\t",ir, region.type, region.x, region.y, region.width, region.height)

    @classmethod
    def find_wnd_area(cls, area_type="VIEW_3D"):
        import bpy
        context = bpy.context
        for iw, window in enumerate(context.window_manager.windows):
            #print(iw, window, window.parent, window.x, window.y, window.width, window.height)
            for ia, area in enumerate(window.screen.areas):
                if area.type == area_type:
                    return area
        return None

    @classmethod
    def find_wnd_region(cls, area_type="VIEW_3D", region_type="UI"):
        import bpy
        context = bpy.context
        for iw, window in enumerate(context.window_manager.windows):
            #print(iw, window, window.parent, window.x, window.y, window.width, window.height)
            for ia, area in enumerate(window.screen.areas):
                if area.type == area_type:
                    #print("\t",ia, area, area.type, area.x, area.y, area.width, area.height)
                    for ir, region in enumerate(area.regions):
                        #print("\t\t",ir, region.type, region.x, region.y, region.width, region.height)
                        if region.type == region_type:
                            return region
        return None
