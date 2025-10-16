def iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Each box is in the format (x1, y1, x2, y2).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def is_inside(inner, outer):
    """
    Returns True if 'inner' box is fully inside 'outer' box.
    """
    return (
        inner[0] >= outer[0] and
        inner[1] >= outer[1] and
        inner[2] <= outer[2] and
        inner[3] <= outer[3]
    )


def check_overlap_area(area: tuple, listOfPaintedBoundingBoxes: list, iou_threshold: float = 0.5) -> bool:
    """
    Checks if the given box overlaps more than the specified IoU threshold
    (default 50%) or is inside / contains any of the existing boxes.

    Returns:
        True  -> if the box overlaps >= threshold or is inside/contains another
        False -> if it's a unique, valid box
    """
    for box in listOfPaintedBoundingBoxes:
        overlap = iou(area, box)

        # Reject if IoU >= 0.5 or either box fully contains the other
        if overlap >= iou_threshold or is_inside(area, box) or is_inside(box, area):
            return True

    return False
