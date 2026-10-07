"""Web adapter for the shared complete survey geometry."""
import numpy as np
from research_bridge import SurveyTour as LegacySurveyTour
from risk_design.closed_tours import ClosedTour, FAMILIES, VERSION

class SurveyTour(LegacySurveyTour):
    def __init__(self,family,radius,center,angle_deg=0):
        self.family=family;self.radius=float(radius);self.center=np.asarray(center,float)
        self.angle=float(angle_deg);self.version=VERSION
        self.core=ClosedTour(family,radius,self.center,np.deg2rad(angle_deg))
        self.length=self.core.length();self.connector_length=self.core.connector_length

    def vessel(self,times,speed,lever):
        s=np.asarray(times)*speed;xy=self.positions(s)
        # Cyclic differences give the same tangent at the closing seam.
        if self.family=='circle_cross':
            delta=self.positions(np.minimum(s+.1,self.length))-self.positions(np.maximum(s-.1,0.))
        else:
            delta=self.positions((s+.1)%self.length)-self.positions((s-.1)%self.length)
        yaw=np.arctan2(delta[:,0],delta[:,1])
        antenna=np.c_[xy,np.zeros(len(xy))]
        offset=np.c_[lever[0]*np.cos(yaw)+lever[1]*np.sin(yaw),-lever[0]*np.sin(yaw)+lever[1]*np.cos(yaw),np.full(len(xy),lever[2])]
        return antenna,antenna+offset


    def preview(self):
        breaks=[x for pair in self.core.transfer_intervals() for x in pair]
        distance=np.unique(np.r_[np.linspace(0.,self.length,801),breaks])
        xy=self.positions(distance)
        midpoint=(distance[:-1]+distance[1:])/2
        transfer=np.r_[~self.core.acquisition_mask(midpoint),False]
        return [dict(x=float(p[0]),y=float(p[1]),transfer=bool(flag)) for p,flag in zip(xy,transfer)]
