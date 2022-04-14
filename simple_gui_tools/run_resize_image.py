import os
import PySimpleGUI as sg

def start_work():
    import cv2
    # import PIL.Image as Image
    
    version = 1.0
    sg.theme('Light Blue 2')
    
    layout = [[sg.Text("功能说明：设置所选目录中所有图片的大小([保存目录]默认为: [图片目录]/format)", text_color="blue"),
               sg.Text("宽"), 
               sg.Input(key="width", size=(4, 1), default_text="512"), 
               sg.Text("高"), 
               sg.Input(key="height", size=(4, 1), default_text="512")],
              [sg.Text("图片目录", auto_size_text=True, justification='left'),
               sg.InputText(key="img_folder", size=(80, 1)), 
               sg.FolderBrowse(key="sel_image_folder", initial_folder=os.path.dirname(__file__), enable_events=True)], 
              [sg.Text("保存目录", auto_size_text=True, justification='left'),
               sg.InputText(key="sav_folder", size=(80, 1)), 
               sg.FolderBrowse(key="sel_save_folder", initial_folder=os.path.dirname(__file__))], 
              [sg.Output(key="console", size=(86, 5))],
              [sg.Button('OK'), sg.Button('Exit')]]
    
    window = sg.Window('设置图片大小工具({})'.format(version), layout)
    
    while True:
        event, values = window.read()
        # print(f'Event: {event}, Values: {values}')
        
        if event == 'OK':
            img_folder = values["img_folder"]
            sav_folder = values["sav_folder"]
            width = values["width"]
            height = values["height"]
            
            if (img_folder == ""):
                print('请选择目录！')
            elif not os.path.exists(img_folder):
                print('目录不存在，请重新选择！')
            elif width == "":
                print("请输入图片宽度")
            elif height == "":
                print("请输入图片高度")
            else:
                print("开始分析...")
                
                if sav_folder == "" or not os.path.exists(sav_folder):
                    sav_folder = "{}/format".format(img_folder)
                    window["sav_folder"].update(sav_folder)
                    print("save path %s" % sav_folder)
                
                tmp_folder = img_folder
                if not os.path.exists(sav_folder):
                    os.mkdir(sav_folder)
                img_folder = tmp_folder
                
                for file in sorted(os.listdir(img_folder)):
                    if file.startswith("."):
                        continue
                    
                    img_path = "{}/{}".format(img_folder, file)
                    sav_path = "{}/{}".format(sav_folder, file)
                    
                    if os.path.isdir(img_path):
                        continue
                    
                    # opencv2实现
                    img = cv2.imread(img_path)
                    resized_img = cv2.resize(img, (int(width), int(height)))
                    cv2.imwrite(sav_path, resized_img)
                    
                    # # pillow实现
                    # im = Image.open(img_path)
                    # out = im.resize((int(width), int(height)), Image.ANTIALIAS)
                    # rgb = out.convert('RGB')
                    # rgb.save(sav_path)

                print("分析结束!!!")
        elif event in (None, 'Exit'):
            break
    window.close()

    
if __name__ == "__main__":
    start_work()
