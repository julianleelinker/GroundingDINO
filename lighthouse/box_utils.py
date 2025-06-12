import torch


def xywh_to_xyxy(bboxes):
    results = bboxes.clone()
    results[:, :2] -= results[:, 2:] / 2
    results[:, 2:] += results[:, :2]
    return results

def xyxy_to_xywh(bboxes):
    results = bboxes.clone()
    results[:, 2:] -= results[:, :2]
    results[:, :2] += results[:, 2:] / 2
    return results

def fix_boundary(bboxes, image_size=(1.0, 1.0)):
    H, W = image_size
    results = bboxes.clone()
    results = results / torch.Tensor([W, H, W, H])
    results = xywh_to_xyxy(results)
    results = torch.clamp(results, 0., 1.)
    results = xyxy_to_xywh(results)
    results = results * torch.Tensor([W, H, W, H])
    return results

def compute_intersection_over_union(bboxes1, bboxes2=None):
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
    # ios = intersection_area / area_min # ios
    ios = intersection_area / (area1+area2) # iou
    return ios

def merge_two_bbox(bbox1, bbox2):
    result = torch.max(bbox1, bbox2)
    result[:2] = torch.min(bbox1, bbox2)[:2]
    return result

def merge_by_iou(bboxes, image_size = (1.0, 1.0), threshold = 0.5):
    if len(bboxes) == 0:
        return bboxes, []
    labels = [str(i) for i in range(len(bboxes))]
    bboxes = xywh_to_xyxy(bboxes)
    H, W = image_size
    bboxes = bboxes * torch.Tensor([W, H, W, H])
    while True:
        try:
            ios = compute_intersection_over_union(bboxes)
            ios = ios - 2.0*torch.eye(ios.size(0))
            max_pos = torch.unravel_index(torch.argmax(ios), ios.shape)
            if ios[max_pos]<threshold:
                break
            bboxes[max_pos[0]] = merge_two_bbox(bboxes[max_pos[0]], bboxes[max_pos[1]])
            bboxes = torch.cat((bboxes[:max_pos[1], :], bboxes[max_pos[1]+1:, :]), dim=0)
            # labels[max_pos[0]] = labels[max_pos[0]] + '_' + labels.pop(max_pos[1])
        except IndexError as e:
            import ipdb; ipdb.set_trace()
    bboxes = bboxes / torch.Tensor([W, H, W, H])
    bboxes = xyxy_to_xywh(bboxes)
    return bboxes, labels
        