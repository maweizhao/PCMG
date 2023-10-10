import pickle as pkl
import os
import sys
import numpy as np
import pickle

import ezc3d

LABELS=['_00155:ARIEL', '_00155:LFHD', '_00155:LBHD', '_00155:RFHD', '_00155:RBHD', '_00155:C7', '_00155:T10', '_00155:CLAV', '_00155:STRN', '_00155:LFSH', '_00155:LBSH', '_00155:LUPA', '_00155:LELB', '_00155:LIEL', '_00155:LFRM', '_00155:LIWR', '_00155:LOWR', '_00155:LIHAND', '_00155:LOHAND', '_00155:RFSH', '_00155:RBSH', '_00155:RUPA', '_00155:RELB', '_00155:RIEL', '_00155:RFRM', '_00155:RIWR', '_00155:ROWR', '_00155:RIHAND', '_00155:ROHAND', '_00155:LFWT', '_00155:MFWT', '_00155:RFWT', '_00155:LBWT', '_00155:MBWT', '_00155:RBWT', '_00155:LTHI', '_00155:LKNE', '_00155:LKNI', '_00155:LSHN', '_00155:LANK', '_00155:LHEL', '_00155:LMT5', '_00155:LMT1', '_00155:LTOE', '_00155:RTHI', '_00155:RKNE', '_00155:RKNI', '_00155:RSHN', '_00155:RANK', '_00155:RHEL', '_00155:RMT5', '_00155:RMT1', '_00155:RTOE']


def write_mocap_c3d(markers: np.ndarray, out_mocap_fname: str,labels: list = LABELS,  frame_rate: int = 120) -> None:
    """

    :param markers: np.ndarray: num_frames x num_points x 3
    :param labels: list: num_points
    :param out_mocap_fname: str:
    :param frame_rate:
    """
    # todo: add the ability to write at any scale. alternatively make it standard to mm
    assert out_mocap_fname.endswith('.c3d')

    writer = ezc3d.c3d()

    writer['parameters']['POINT']['RATE']['value'] = [frame_rate]
    writer['parameters']['POINT']['LABELS']['value'] = labels
    
    #maweizhaochange!
    #markers = markers * 1000. # meters to milimeters. a c3d file should be in mm

    pts = markers
    pts_extra = np.zeros([markers.shape[0], markers.shape[1], 1])
    points = np.concatenate([pts, pts_extra], axis=-1).astype(float)

    nan_mask = (np.logical_or(pts == 0, np.isnan(pts))).sum(-1) == 3
    nan_mask_repeated = np.repeat(nan_mask[:, :, None], repeats=4, axis=-1)
    points[nan_mask_repeated] = np.nan

    residuals = np.ones(points.shape[:-1])
    residuals[nan_mask] = -1
    residuals = residuals[:, :, None]

    writer['data']['points'] = points.transpose([2, 1, 0])

    writer['data']['meta_points']['residuals'] = residuals.transpose([2, 1, 0])
    writer.write(out_mocap_fname)


def read_mocap(mocap_fname):
    labels = None
    frame_rate = None
    if mocap_fname.endswith('.mat'):
        import scipy.io
        _marker_data = scipy.io.loadmat(mocap_fname)

        markers = None
        expected_marker_data_fields = ['MoCaps', 'Markers']
        for expected_key in expected_marker_data_fields:
            if expected_key in _marker_data.keys():
                markers = _marker_data[expected_key]
        if markers is None:
            raise ValueError(
                f"The .mat file do not have the expected field for marker data! Expected fields are {expected_marker_data_fields}")
        if 'Labels' in _marker_data.keys():
            labels = np.vstack(_marker_data['Labels'][0]).ravel()

    elif mocap_fname.endswith('.pkl'):
        with open(mocap_fname, 'rb') as f:
            _marker_data = pickle.load(f, encoding='latin-1')
        markers = _marker_data['markers']
        if 'required_parameters' in _marker_data.keys():
            frame_rate = _marker_data['required_parameters']['frame_rate']
        elif 'frame_rate' in _marker_data:
            frame_rate = _marker_data['frame_rate']
        labels = _marker_data.get('labels', False)
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        # address a bug in bmlmovi
        labels = [f'*{lid}' if isinstance(l, np.ndarray) else l for lid, l in enumerate(labels)]

    elif mocap_fname.endswith('.c3d'):
        _marker_data = ezc3d.c3d(mocap_fname)
        # points_residuals = c['data']['meta_points']['residuals']
        # analog_data = c['data']['analogs']
        markers = _marker_data['data']['points'][:3].transpose(2, 1, 0)
        frame_rate = _marker_data['parameters']['POINT']['RATE']['value'][0]
        labels = _marker_data['parameters']['POINT']['LABELS']['value']
        if len(labels) < markers.shape[1]:
            labels = labels + [f'*{len(labels) + i:d}' for i in range(markers.shape[1] - len(labels))]

    elif mocap_fname.endswith('.npz'):
        _marker_data = np.load(mocap_fname, allow_pickle=True)
        markers = _marker_data['markers']
        if 'frame_rate' in list(_marker_data.keys()):
            frame_rate = _marker_data['frame_rate']
        elif 'required_parameters' in list(_marker_data.keys()):
            if 'frame_rate' in _marker_data['required_parameters']:
                frame_rate = _marker_data['required_parameters']['frame_rate']

        labels = _marker_data.get('labels', None)

    else:
        raise ValueError(f"Error! Could not recognize file format for {mocap_fname}")

    if labels is None:
        labels = [f'*{i}' for i in range(markers.shape[1])]
    elif len(labels) < markers.shape[1]:
        labels = labels + [f'*{i}' for i in range(markers.shape[1] - len(labels))]

    labels = [l.decode() if isinstance(l, bytes) else l for l in labels]

    subject_mask = []
    subject_id_map = {}
    for l in labels:

        subject_name = l.split(':')[0] if ':' in l else 'null'
        if subject_name not in subject_id_map: subject_id_map[subject_name] = len(subject_id_map)
        subject_mask.append(subject_id_map[subject_name])

    subject_mask = {sname: np.array([i == sid for i in subject_mask], dtype=bool) for sname, sid in
                    subject_id_map.items()}

    return {'markers': markers, 'labels': labels, 'frame_rate': frame_rate, '_marker_data': _marker_data,
            'subject_mask': subject_mask}




# path="D:\maweizhao\MyProgram\DeepLearning\myfile\PCMG/temp\c3d\kick_001.c3d"
# data=read_mocap(path)

# write_points=data['markers'][0:2,:,:]
# write_path="./temp/c3d/kick_001_cut1.c3d"
# write_labels=LABELS
# frame_rate=10

# #print(data['labels'])

# write_mocap_c3d(write_points,write_path,write_labels,frame_rate)

