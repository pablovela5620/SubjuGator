from __future__ import division
import tf
import numpy as np
from twisted.internet import defer
from txros import util
from sensor_msgs.msg import RegionOfInterest
from sensor_msgs.msg import CameraInfo
from image_geometry import PinholeCameraModel
from mil_misc_tools import text_effects

fprint = text_effects.FprintFactory(title="PATH_MARKER", msg_color="cyan").fprint

@util.cancellableInlineCallbacks
def run(sub):
    current_pose = sub.pose
    cam_info_sub = yield sub.nh.subscribe('/camera/front/left/camera_info',
                                          CameraInfo)

    fprint('Obtaining cam info message')
    cam_info = yield cam_info_sub.get_next_message()
    model = PinholeCameraModel()
    model.fromCameraInfo(cam_info)
    
    print(sub.pose.orientation)
    get_path_marker = yield sub.nh.subscribe('/path_marker_roi', RegionOfInterest)
    while(True):
        path_marker = yield get_path_marker.get_next_message()
        if (path_marker.x_offset != 0):
            break
    print('Found path marker: {}'.format(path_marker))
    center_x = path_marker.x_offset + path_marker.width/2
    center_y = path_marker.y_offset + path_marker.height/2
    
    ray, pos = yield get_transform(sub, model, np.array([center_x, center_y]))
    print(ray)
    point = LinePlaneCollision(np.array([0, 0, 1]), np.array([0, 0, sub.dvl_range]), ray, pose)
    print('MOVING!!!')
    point[2] = current_pose.position[2]
    yield sub.set_position(point).go(speed=0.2)

    
    

@util.cancellableInlineCallbacks
def get_transform(sub, model, point):
    fprint('Projecting to 3d ray')
    ray = np.array(model.projectPixelTo3dRay(point))
    fprint('Transform')
    transform = yield sub._tf_listener.get_transform('/map', 'front_left_cam')
    ray = transform._q_mat.dot(ray)
    ray = ray / np.linalg.norm(ray)
    marker = Marker(
        ns='dice',
        action=visualization_msgs.Marker.ADD,
        type=Marker.ARROW,
        scale=Vector3(0.2, 0.2, 2),
        points=np.array([
            Point(transform._p[0], transform._p[1], transform._p[2]),
            Point(transform._p[0] + ray[0], transform._p[1] + ray[1],
                  transform._p[2] + ray[2]),
        ]))
    marker.header.frame_id = '/map'
    marker.color.r = 0
    marker.color.g = 1
    marker.color.b = 0
    marker.color.a = 1
    global pub_cam_ray
    pub_cam_ray.publish(marker)
    fprint('ray: {}, origin: {}'.format(ray, transform._p))
    defer.returnValue((ray, transform._p))

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
 
	ndotu = planeNormal.dot(rayDirection)
	if abs(ndotu) < epsilon:
		raise RuntimeError("no intersection or line is within plane")
 
	w = rayPoint - planePoint
	si = -planeNormal.dot(w) / ndotu
	Psi = w + si * rayDirection + planePoint
	return Psi
