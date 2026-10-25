"""Continuous-acceleration, bounded-jerk timing for joint action waypoints."""
import numpy as np
from scipy.interpolate import BPoly, PchipInterpolator


def continuous_curve(times, values):
    times=np.asarray(times,float);values=np.asarray(values,float)
    dt=np.diff(times)[:,None]
    secants=np.abs(np.diff(values,axis=0))/dt
    slopes=PchipInterpolator(times,values).derivative()(times)
    # Keep every Bernstein control point inside its segment endpoints. This
    # guarantees no joint-position overshoot while retaining through velocities.
    limits=2.5*np.minimum(secants[:-1],secants[1:])
    slopes[1:-1]=np.clip(slopes[1:-1],-limits,limits)
    slopes[[0,-1]]=0
    zero=np.zeros(values.shape[1])
    return BPoly.from_derivatives(times,[[q,v,zero] for q,v in zip(values,slopes)])


def derivative_bounds(curve):
    """Conservative per-interval bounds, by Bernstein's convex-hull property."""
    return [np.max(np.abs(curve.derivative(order).c),axis=(0,2)) for order in (1,2,3)]


def retime_continuous(q,h,times,speed_scale):
    times=np.asarray(times,float).copy()
    for _ in range(60):
        factors=np.ones(len(times)-1)
        for values,limits in ((q,(.15,.3,.6)),(h,(.35,.7,1.4))):
            for order,(bounds,limit) in enumerate(zip(derivative_bounds(continuous_curve(times,values)),limits),1):
                factors=np.maximum(factors,(bounds/limit)**(1/order))
        if np.max(factors)<=1+1e-8:return times/speed_scale
        times=np.r_[0,np.cumsum(np.diff(times)*np.where(factors>1,factors*1.01,1.))]
    raise ValueError('Continuous action timing failed to converge; no rollout authorized.')


def check_continuous(q,h,times,speed_scale):
    curves=[continuous_curve(times,values) for values in (q,h)]
    for curve,limits in zip(curves,((.15,.3,.6),(.35,.7,1.4))):
        for order,(bounds,limit) in enumerate(zip(derivative_bounds(curve),limits),1):
            if np.max(bounds)>limit*speed_scale**order+1e-6:
                raise ValueError('Continuous action exceeds speed, acceleration or jerk limit.')
    return curves


def simplify_actions(q,h,times,arm_tolerance=.003,hand_tolerance=.01):
    """Remove redundant time-aligned waypoints within explicit joint tolerances."""
    values=np.c_[q/arm_tolerance,h/hand_tolerance]
    keep={0,len(times)-1};pending=[(0,len(times)-1)]
    while pending:
        first,last=pending.pop()
        if last-first<2:continue
        weight=(times[first+1:last]-times[first])/(times[last]-times[first])
        linear=values[first]+weight[:,None]*(values[last]-values[first])
        error=np.max(np.abs(values[first+1:last]-linear),axis=1)
        index=int(np.argmax(error))+first+1
        if error[index-first-1]>1:
            keep.add(index);pending.extend([(first,index),(index,last)])
    return np.array(sorted(keep),dtype=int)
