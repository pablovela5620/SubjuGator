from ros_alarms import HandlerBase, HeartbeatMonitor
from std_msgs.msg import Header


class NetworkLoss(HandlerBase):
    alarm_name = 'network-loss'

    def __init__(self):
        self.hm = HeartbeatMonitor(self.alarm_name, "/network", Header, prd=0.8, node_name="network_loss_kill")

    def raised(self, alarm):
        pass

    def cleared(self, alarm):
        pass
