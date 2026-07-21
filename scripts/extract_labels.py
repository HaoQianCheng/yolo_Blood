"""从 render 标注图提取标签框，生成 YOLO-OBB 旋转框格式标注文件

原理：origin 和 render 背景几乎一致，差异处即为人工标注线。
     用 absdiff 取差异图，找轮廓，识别"外层大框"和"里面的标签框"。
     标签框 = 竖长方形(顶点≈4)，完全被外层大框包含。
     超出大框的图丢弃(标注异常)。
     用 minAreaRect 取旋转框 4 角点，输出 YOLO-OBB 格式:
     class x1 y1 x2 y2 x3 y3 x4 y4 (绝对像素坐标)
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import cv2
import numpy as np

# absdiff 二值化阈值：差异大于此值即视为标注线
DIFF_THRESHOLD = 15
# 闭运算 kernel 和次数：连接颜色不均/断续的框线
CLOSE_KERNEL = 7
CLOSE_ITERS = 3
# 候选轮廓最小面积(像素)，过滤碎块
MIN_CONTOUR_AREA = 500

# 外层大框特征：面积占比大(>30%图面积)、近正方形或横长方形(h/w<1.2)
OUTER_AREA_RATIO_MIN = 0.30
OUTER_HW_MAX = 1.2

# 标签框特征：竖长方形(h/w 在此范围)、面积 ≥ 下限(排除碎块)、完全在大框内
# 不卡顶点数：框线颜色不均匀时 approxPolyDP 顶点会 >4，真框会被误排除
LABEL_HW_MIN = 1.2
LABEL_HW_MAX = 2.5
LABEL_MIN_AREA = 30000

CLASS_ID = 0


def _get_contours(origin: np.ndarray, render: np.ndarray):
    """absdiff + 二值化 + 闭运算 + 找轮廓，返回候选轮廓列表(带属性，按面积降序)"""
    h, w = render.shape[:2]
    diff = cv2.absdiff(origin, render)
    diff_max = diff.max(axis=2)
    _, binarized = cv2.threshold(diff_max, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)
    closed = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERS)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cands = []
    img_area = w * h
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, bw, bh = cv2.boundingRect(c)
        cands.append({
            "area": area, "cnt": c, "x": x, "y": y, "w": bw, "h": bh,
            "vertices": len(approx), "hw": bh / max(bw, 1),
            "area_ratio": area / img_area,
        })
    cands.sort(key=lambda v: v["area"], reverse=True)
    return cands, (w, h)


def _corners_inside(outer: tuple, corners) -> bool:
    """outer = (x, y, w, h)；corners 是 minAreaRect 的 4 角点。
    所有角点都在 outer 内才返回 True。用于精确检测倾斜框角点是否超出大框。"""
    ox, oy, ow, oh = outer
    for px, py in corners:
        if not (ox <= px <= ox + ow and oy <= py <= oy + oh):
            return False
    return True


def extract_label_obb(render_path: str, origin_path: str = None) -> tuple[list[tuple[int, int]], int] | None:
    """从差异图中提取标签旋转框。

    Args:
        render_path: 标注图路径
        origin_path: 原图路径，None 则自动匹配同名 origin 文件

    Returns:
        (corners, class_id) 其中 corners 是 4 个角点 [(x,y),...] 绝对像素坐标；
        或 None 表示未检测到/丢弃(标签超出大框)
    """
    render = cv2.imread(str(render_path))
    if render is None:
        raise FileNotFoundError(f"无法读取图片: {render_path}")

    if origin_path is None:
        origin_path = str(render_path).replace(".render.jpg", ".origin.jpg")
    origin = cv2.imread(str(origin_path))
    if origin is None:
        raise FileNotFoundError(f"无法读取原图: {origin_path}")

    h, w = render.shape[:2]
    if origin.shape[:2] != (h, w):
        raise ValueError(f"原图和标注图尺寸不一致: {origin.shape[:2]} vs {(h, w)}")

    cands, (w, h) = _get_contours(origin, render)
    if not cands:
        return None

    # 1) 识别外层大框：面积占比>30% 且 h/w<1.2 的最大那个
    outer = None
    for c in cands:
        if c["area_ratio"] > OUTER_AREA_RATIO_MIN and c["hw"] < OUTER_HW_MAX:
            outer = (c["x"], c["y"], c["w"], c["h"])
            break
    if outer is None:
        return None

    # 2) 识别标签框：竖长方形(h/w 1.2~2.5)、面积≥下限、4角点全在大框内
    #    不卡顶点数(框线不均时顶点>4)；取面积最大的合格候选
    label = None
    for c in cands:
        if c["area_ratio"] > OUTER_AREA_RATIO_MIN:
            continue  # 排除外层大框本身
        if c["area"] < LABEL_MIN_AREA:
            continue  # 排除小碎块
        if not (LABEL_HW_MIN <= c["hw"] <= LABEL_HW_MAX):
            continue
        # 用 minAreaRect 4 角点精确判断是否超出大框(支持倾斜)
        mrect = cv2.minAreaRect(c["cnt"])
        box = cv2.boxPoints(mrect)
        if not _corners_inside(outer, box):
            continue  # 标签框角点超出大框 → 跳过(严格丢弃)
        label = c
        break  # cands 按面积降序，第一个合格的就是最大的

    if label is None:
        return None

    # 3) minAreaRect 取旋转框 4 角点(支持倾斜)
    mrect = cv2.minAreaRect(label["cnt"])
    box = cv2.boxPoints(mrect)
    corners = [(int(round(p[0])), int(round(p[1]))) for p in box]
    return (corners, CLASS_ID)


def _format_obb(corners: list[tuple[int, int]], cls: int) -> str:
    """YOLO-OBB 行: class x1 y1 x2 y2 x3 y3 x4 y4"""
    flat = " ".join(f"{x} {y}" for x, y in corners)
    return f"{cls} {flat}"


def match_image_pairs(raw_dir: str) -> list[tuple[Path, Path]]:
    """匹配原图和标注图对。"""
    raw = Path(raw_dir)
    origins = sorted(raw.glob("*.origin.jpg"))
    renders = sorted(raw.glob("*.render.jpg"))

    if len(origins) != len(renders):
        print(f"警告: 原图 {len(origins)} 张 ≠ 标注图 {len(renders)} 张")

    pairs = list(zip(origins, renders))
    print(f"匹配到 {len(pairs)} 对图片")
    return pairs


def _process_one(args: tuple) -> str:
    """处理一对图，写 txt。args=(origin_str, render_str, out_dir_str)。
    返回状态: 'ok' / 'empty' / 'error:...'。供进程池 worker 调用。"""
    origin_str, render_str, out_dir_str = args
    origin_path = Path(origin_str)
    render_path = Path(render_str)
    out_dir = Path(out_dir_str)
    try:
        result = extract_label_obb(str(render_path), str(origin_path))
    except Exception as e:
        return f"error:{origin_path.name}:{e}"

    txt_path = out_dir / origin_path.name.replace(".origin.jpg", ".txt")
    if result is None:
        txt_path.write_text("")
        return f"empty:{origin_path.name}"

    corners, cls = result
    txt_path.write_text(_format_obb(corners, cls) + "\n")
    return f"ok:{origin_path.name}"


def process_all(raw_dir: str, output_dir: str = None, workers: int = None) -> None:
    """批量并行处理：为每张原图生成 YOLO-OBB 标注 txt。"""
    if output_dir is None:
        output_dir = raw_dir
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pairs = match_image_pairs(raw_dir)
    if not pairs:
        return

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    tasks = [(str(o), str(r), str(out)) for o, r in pairs]
    success = empty = error = 0
    done = 0
    total = len(tasks)

    print(f"并行处理: {total} 对, {workers} 进程")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process_one, t): t for t in tasks}
        for fut in as_completed(futures):
            status = fut.result()
            done += 1
            if status.startswith("ok"):
                success += 1
            elif status.startswith("empty"):
                empty += 1
                print(f"丢弃/无框: {status.split(':', 1)[1]}")
            else:
                error += 1
                print(f"错误: {status}")
            if done % 50 == 0 or done == total:
                print(f"  进度 {done}/{total} (有框 {success}, 丢弃 {empty}, 错误 {error})")

    print(f"\n处理完成: 有框 {success} 张, 丢弃/无框 {empty} 张, 错误 {error} 张")


def main():
    parser = argparse.ArgumentParser(description="从标注图提取 YOLO-OBB 旋转框标注")
    parser.add_argument("--raw", default="data/raw", help="原始图片目录")
    parser.add_argument("--output", default=None, help="标注输出目录（默认同 --raw）")
    parser.add_argument("--workers", type=int, default=None, help="并行进程数（默认 CPU核数-1）")
    args = parser.parse_args()
    process_all(args.raw, args.output, args.workers)


if __name__ == "__main__":
    main()
