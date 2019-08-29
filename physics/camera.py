import pybullet as p


class Camera(object):

    def __init__(self,
                 target_pos=(0, 0, 0),
                 pitch=-30.0,
                 yaw=60,
                 roll=0,
                 cam_dist=20,
                 width=480,
                 height=320,
                 up_axis=2,
                 near_plane=0.01,
                 far_plane=100,
                 fov=60):
        self.target_pos = target_pos
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.cam_dist = cam_dist
        self.width = width
        self.height = height
        self.up_axis = up_axis
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.fov = fov

        self.view_mat = p.computeViewMatrixFromYawPitchRoll(target_pos, cam_dist, yaw, pitch, roll, up_axis)
        aspect = width / height
        self.proj_mat = p.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)

    def get_params(self):
        params = {
            'target_pos': self.target_pos,
            'pitch': self.pitch,
            'yaw': self.yaw,
            'roll': self.roll,
            'cam_dist': self.cam_dist,
            'width': self.width,
            'height': self.height,
            'up_axis': self.up_axis,
            'near_plane': self.near_plane,
            'far_plane': self.far_plane,
            'fov': self.fov
        }
        return params

    def take_pic(self):
        img_arr = p.getCameraImage(self.width, self.height, self.view_mat, self.proj_mat)
        return img_arr[2]

    def take_seg(self):
        img_arr = p.getCameraImage(self.width, self.height, self.view_mat, self.proj_mat)
        return img_arr[4]
