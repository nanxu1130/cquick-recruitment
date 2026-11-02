import cv2
import numpy as np
import os
from datetime import datetime

NAME_PINYIN = "liulutong"
MIN_AREA = 500

def get_red_mask(hsv):
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return mask1 + mask2

def get_blue_mask(hsv):
    # 任务2：放宽蓝色阈值（对比实验）
    lower_blue = np.array([90, 60, 60])
    upper_blue = np.array([140, 255, 255])
    return cv2.inRange(hsv, lower_blue, upper_blue)

def main():
    print(f"姓名：{NAME_PINYIN}")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("【任务2：参数对比实验（放宽蓝色阈值）】")
    print("="*40)

    ROOT_DIR = r"D:\Week2_Assignment\liulutong"
    img_dir = os.path.join(ROOT_DIR, "data", "images")
    output_dir = os.path.join(ROOT_DIR, "outputs", "task2")  # ✅ 改到新建的 task2
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"[调试] 图片读取目录: {img_dir}")
    print(f"[调试] 结果保存目录: {output_dir}")
    print("-"*40)

    img_names = ["task_blue_small.jpg", "task_red.jpg", "task_blue_hard.jpg"]
    
    for name in img_names:
        img_path = os.path.join(img_dir, name)
        if not os.path.exists(img_path):
            print(f"[警告] 未找到：{img_path}")
            continue
            
        img = cv2.imread(img_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        if "red" in name:
            mask = get_red_mask(hsv)
            color_name = "红色"
        else:
            mask = get_blue_mask(hsv)
            color_name = "蓝色"
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"处理：{name} | 候选轮廓数量：{len(contours)}")
        
        if contours:
            max_cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_cnt)
            if area > MIN_AREA:
                M = cv2.moments(max_cnt)
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])
                print(f"  → 检测到{color_name}！u={u}, v={v}, area={int(area)}")
                
                cv2.drawContours(img, [max_cnt], -1, (0,255,0), 2)
                cv2.circle(img, (u,v), 5, (0,0,255), -1)
                cv2.putText(img, f"u={u},v={v}", (u+10, v-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                
                save_name = f"result_task_{os.path.splitext(name)[0]}.png"
                save_path = os.path.join(output_dir, save_name)
                cv2.imwrite(save_path, img)
                print(f"  → 结果已保存：{save_path}\n")
            else:
                print(f"  → 面积过小（{int(area)} < {MIN_AREA}），忽略\n")
        else:
            print(f"  → 未检测到轮廓\n")
            
    print("="*40)
    print("任务2参数对比实验完成！图片已存入 outputs/task2/")
    print("="*40)

if __name__ == "__main__":
    main()
