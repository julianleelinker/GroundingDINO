import torch


'''
    overlap_x0 = max(bbox1[0], bbox2[0])
    overlap_y0 = max(bbox1[1], bbox2[1])
    overlap_x1 = min(bbox1[2], bbox2[2])
    overlap_y1 = min(bbox1[3], bbox2[3])
    overlap_width = max(0, overlap_x1 - overlap_x0)
    overlap_height = max(0, overlap_y1 - overlap_y0)
    small_area = min((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]), (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))
    return overlap_width * overlap_height / small_area 
'''


# t1 = torch.tensor()
# create torch tensor with 3, 4 shape, and fill with 1

def compute_intersection_over_small(bboxes1, bboxes2=None):
    if bboxes2 is None:
        bboxes2 = bboxes1.clone()

    area1 = ((bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])).unsqueeze(-1)
    area2 = ((bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])).unsqueeze(0)
    area_min = torch.min(area1, area2)

    # add dummy dimension 1 to bbox1
    bboxes1 = bboxes1.unsqueeze(-1)
    # add dummy dimension 1 to bbox2 to last dimension
    bboxes2 = bboxes2.unsqueeze(-1)
    # reverse dimension of bbox2
    bboxes2 = bboxes2.permute(2, 1, 0)

    ma = torch.max(bboxes1, bboxes2)
    mi = torch.min(bboxes1, bboxes2)
    width = torch.max(torch.tensor([0]), mi[:, 2, :] - ma[:, 0, :])
    height = torch.max(torch.tensor([0]), mi[:, 3, :] - ma[:, 1, :])
    intersection_area = width * height
    ios = intersection_area / area_min
    return ios