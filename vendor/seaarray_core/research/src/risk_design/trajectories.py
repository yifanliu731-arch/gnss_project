"""Complete C1 tours with explicit common-pose connectors; no budget prefixes.

Loop scale is the maximum distance from the array centroid. Connectors are
counted separately and can extend beyond that envelope. Curvature feasibility
is numerical (dense checks), not an assertion about an actual vessel controller.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from risk_design.geometry import arc_segment, line_segment, concat, path_length, pose_at


def fillet_polygon(vertices,radius):
    vertices=np.asarray(vertices)
    incoming=vertices-np.roll(vertices,1,axis=0)
    outgoing=np.roll(vertices,-1,axis=0)-vertices
    lengths=np.linalg.norm(outgoing,axis=1)
    hi=np.arctan2(incoming[:,1],incoming[:,0]);ho=np.arctan2(outgoing[:,1],outgoing[:,0])
    turn=np.arctan2(np.sin(ho-hi),np.cos(ho-hi))
    cut=radius*np.abs(np.tan(turn/2))
    if np.any(cut+np.roll(cut,-1)>=lengths):raise ValueError('fillets overlap: insufficient scale for turning radius')
    enter=vertices-incoming/np.linalg.norm(incoming,axis=1)[:,None]*cut[:,None]
    leave=vertices+outgoing/lengths[:,None]*cut[:,None]
    parts=[]
    for i in range(len(vertices)):
        parts.append(arc_segment(*enter[i],hi[i],radius,turn[i],label='survey_turn'))
        parts.append(line_segment(*leave[i],*enter[(i+1)%len(vertices)],h0=ho[i],label='survey'))
    return concat(parts)


def two_line_cross(center, radius, angle):
    """Two straight diameters through the center. Not a closed tour."""
    from risk_design.geometry import Segment
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    local = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    points = local @ rotation.T * float(radius) + np.asarray(center, dtype=float)
    parts = []
    for start, end in zip(points[:-1], points[1:]):
        dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
        length = float(np.hypot(dx, dy))
        heading = float(np.arctan2(dy, dx))
        parts.append(Segment(
            "line", "survey", length,
            float(start[0]), float(start[1]), heading,
            float(end[0]), float(end[1]), heading,
        ))
    return parts


def straight_polygon(vertices):
    """Closed straight sides. Corners may change heading without a minimum radius."""
    from risk_design.geometry import Segment
    vertices=np.asarray(vertices,dtype=float)
    parts=[]
    for i in range(len(vertices)):
        a=vertices[i];b=vertices[(i+1)%len(vertices)]
        dx,dy=float(b[0]-a[0]),float(b[1]-a[1]);length=float(np.hypot(dx,dy))
        if length<1e-6:continue
        heading=float(np.arctan2(dy,dx))
        parts.append(Segment('line','survey',length,float(a[0]),float(a[1]),heading,float(b[0]),float(b[1]),heading))
    if not parts:raise ValueError('empty straight polygon')
    return parts


def connector(start, end, radius=280.):
    """Shortest of the two same-turn tangent connectors (LSL, RSR).

    radius<=0 keeps a straight connector and does not impose a turning constraint.
    """
    if radius<=0.:
        from risk_design.geometry import Segment
        x0,y0,x1,y1=float(start[0]),float(start[1]),float(end[0]),float(end[1])
        dx,dy=x1-x0,y1-y0;length=float(np.hypot(dx,dy))
        heading=float(np.arctan2(dy,dx)) if length>1e-9 else float(end[2])
        return [Segment('line','connector',length,x0,y0,heading,x1,y1,float(end[2]))]
    alternatives=[]
    for side in (1.,-1.):
        a=np.array(start[:2])+side*radius*np.array([-np.sin(start[2]),np.cos(start[2])])
        b=np.array(end[:2])+side*radius*np.array([-np.sin(end[2]),np.cos(end[2])])
        h=np.arctan2(*(b-a)[::-1])
        turn1=side*((side*(h-start[2]))%(2*np.pi))
        p=arc_segment(*start,radius,turn1,label='connector')
        n=side*radius*np.array([-np.sin(h),np.cos(h)])
        target=b-n
        q=line_segment(p.x1,p.y1,*target,h0=p.h1,label='connector')
        turn2=side*((side*(end[2]-h))%(2*np.pi))
        r=arc_segment(q.x1,q.y1,q.h1,radius,turn2,label='connector')
        parts=concat([p,q,r])
        if np.linalg.norm(np.array([r.x1,r.y1])-end[:2])>1e-5:
            raise ValueError('connector endpoint mismatch')
        alternatives.append(parts)
    return min(alternatives,key=path_length)


class CompleteTour:
    def __init__(self,family,radius,center,angle=0.,turn_radius=280.,resolution=16385):
        u=np.linspace(0,2*np.pi,65)
        if family=='circle': xy=np.c_[np.cos(u),np.sin(u)]
        elif family=='figure_eight': xy=np.c_[np.sin(u),.5*np.sin(2*u)]
        elif family=='rose':
            # GARNO four-petal rose rho=A cos(2u). Shape baseline only, not Xue's criterion.
            th=np.linspace(0,2*np.pi,257)
            rho=np.cos(2*th);xy=np.c_[rho*np.cos(th),rho*np.sin(th)];xy[-1]=xy[0]
        elif family=='clover':
            # Three wedge-shaped petals, following the observed SW/NW/SE order.
            # Idealized shape only; periodic cubic smoothing is explicitly synthetic.
            xy=np.array([[0,-.55],[-3.8,-4.5],[0,-5.6],[0,-.55],
                         [0,4.6],[-3.5,3.0],[0,-.55],[4.,-4.2],[5.2,-.6],[0,-.55]])
        elif family=='square':xy=np.array([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1],[1,0]])
        elif family=='eight_arm':
            arms=8
            a=np.arange(arms*2)*np.pi/arms
            r=np.where(np.arange(arms*2)%2==0,1.,.23)
            xy=np.c_[r*np.cos(a),r*np.sin(a)];xy=np.vstack([xy,xy[0]])
        elif family=='cross':
            raise ValueError('two straight lines are not a standalone tour')
        elif family in ('circle_cross','square_cross'):
            primary='circle' if family=='circle_cross' else 'square'
            first=CompleteTour(primary,radius,center,angle,turn_radius,resolution)
            lines=two_line_cross(center,radius,angle)
            line_length=path_length(lines)
            start=(*first.curve(0.),float(np.arctan2(*first.curve(0.,1)[::-1]))) if first.poly is None else pose_at(first.poly,0.)
            gap=connector(start,pose_at(lines,0.),turn_radius)
            self.poly=None;self.curve=None
            self.mid=[(first,first.loop_length),(gap,path_length(gap)),(lines,line_length)]
            self.loop_length=float(sum(item[1] for item in self.mid));self.min_radius=first.min_radius
            self.home=(*center,0.);self.inbound=first.inbound
            self.outbound=connector(pose_at(lines,line_length),self.home,turn_radius)
            self.connector_length=path_length(self.inbound)+path_length(self.outbound)
            self.family=family;self.radius=radius;self.angle=angle
            return
        else:raise ValueError(family)
        xy[-1]=xy[0]
        self.poly=None
        self.mid=None
        if family not in ('circle','figure_eight','rose'):
            rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            vertices=xy[:-1]@rot.T
            base=radius/np.linalg.norm(vertices,axis=1).max()
            for _ in range(8):
                if turn_radius>0.:
                    self.poly=fillet_polygon(vertices*base+center,turn_radius)
                else:
                    self.poly=straight_polygon(vertices*base+center)
                length=path_length(self.poly)
                p=np.array([pose_at(self.poly,s)[:2] for s in np.linspace(0,length,2049)])
                envelope=np.linalg.norm(p-center,axis=1).max()
                base*=radius/envelope
            self.loop_length=path_length(self.poly);self.min_radius=0. if turn_radius<=0. else turn_radius
            pose=pose_at(self.poly,0)
            self.home=(*center,0.)
            self.inbound=connector(self.home,pose,turn_radius)
            self.outbound=connector(pose,self.home,turn_radius)
            self.connector_length=path_length(self.inbound)+path_length(self.outbound)
            self.family=family;self.radius=radius;self.angle=angle
            self.mid=None
            return
        knot=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        knot/=knot[-1]
        curve=CubicSpline(knot,xy,bc_type='periodic')
        dense=np.linspace(0,1,resolution)
        unit=curve(dense);scale=radius/np.linalg.norm(unit,axis=1).max()
        rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        self.curve=CubicSpline(knot,xy@rot.T*scale+center,bc_type='periodic')
        d=self.curve(dense,1);dd=self.curve(dense,2)
        speed=np.linalg.norm(d,axis=1)
        if np.min(speed)<1e-8:raise ValueError('stationary spline point')
        curvature=np.abs(d[:,0]*dd[:,1]-d[:,1]*dd[:,0])/speed**3
        self.min_radius=1/max(curvature)
        if turn_radius>0. and self.min_radius<turn_radius*(1-1e-4):
            raise ValueError(f'curvature: minimum radius {self.min_radius:.3f} < {turn_radius}')
        self.u=dense;self.s=cumulative_trapezoid(speed,dense,initial=0.)
        self.loop_length=float(self.s[-1]);point=self.curve(0.);tangent=self.curve(0.,1)
        pose=(*point,float(np.arctan2(tangent[1],tangent[0])))
        self.home=(*center,0.)
        self.inbound=connector(self.home,pose,turn_radius)
        self.outbound=connector(pose,self.home,turn_radius)
        self.connector_length=path_length(self.inbound)+path_length(self.outbound)
        self.family=family;self.radius=radius;self.angle=angle
        self.mid=None

    def _loop_point(self,local):
        local=float(local)%self.loop_length
        if not self.mid:
            return pose_at(self.poly,local)[:2] if self.poly is not None else self.curve(np.interp(local,self.s,self.u))
        for piece,length in self.mid:
            if local<=length+1e-8:
                if hasattr(piece,'poly') and piece.poly is not None:return pose_at(piece.poly,min(local,length))[:2]
                if hasattr(piece,'curve') and piece.curve is not None:return piece.curve(np.interp(min(local,piece.loop_length),piece.s,piece.u))
                return pose_at(piece,min(local,length))[:2]
            local-=length
        raise ValueError('loop coordinate past composite')

    def length(self,repeats=1):return self.connector_length+repeats*self.loop_length

    def positions(self,distance,repeats=1):
        distance=np.atleast_1d(distance);out=np.empty((len(distance),3))
        li=path_length(self.inbound);end=li+repeats*self.loop_length
        for i,s in enumerate(distance):
            if s<0 or s>self.length(repeats)+1e-5:raise ValueError('outside complete tour')
            if s<li:out[i,:2]=pose_at(self.inbound,s)[:2]
            elif s<=end:
                out[i,:2]=self._loop_point((s-li)%self.loop_length)
            else:out[i,:2]=pose_at(self.outbound,s-end)[:2]
        out[:,2]=-10.
        return out

    def schedule(self,repeats=1,speed=1.8,slot_s=12.):
        duration=self.length(repeats)/speed
        # A full slot, including reply/listen guard, must finish before returning.
        times=np.arange(0,max(0.,duration-slot_s)+1e-9,slot_s)
        return times,self.positions(times*speed,repeats),np.resize([2,5,6],len(times))

    def audit(self,repeats=1,speed=1.8):
        duration=self.length(repeats)/speed
        time=np.r_[np.arange(0,duration,2.),duration]
        xyz=self.positions(time*speed,repeats)
        return dict(family=self.family,radius_m=self.radius,angle_rad=self.angle,
                    repeats=repeats,loop_length_m=self.loop_length,connector_length_m=self.connector_length,
                    total_seconds=duration,min_loop_turn_radius_m=self.min_radius,
                    closure_m=float(np.linalg.norm(xyz[-1]-xyz[0])),
                    endpoint_heading_error_rad=float(abs(np.arctan2(np.sin(self.outbound[-1].h1),np.cos(self.outbound[-1].h1)))),
                    actual_envelope_m=float(np.linalg.norm(xyz[:,:2]-self.home[:2],axis=1).max()),
                    time=time,ship=xyz)
