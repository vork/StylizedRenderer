"""Frame-rate independent fly camera and continuous day/night clock."""
import math
import numpy as np
from .math3d import normalize


class FlyCamera:
    def __init__(self,eye,target):
        self.eye=np.asarray(eye,dtype=np.float32).copy()
        direction=normalize(np.asarray(target)-self.eye)
        self.yaw=math.atan2(float(direction[0]),float(-direction[2]))
        self.pitch=math.asin(float(np.clip(direction[1],-1,1)))
        self.speed=max(float(np.linalg.norm(np.asarray(target)-self.eye))*.15,.3)

    @property
    def forward(self):
        return np.array([math.sin(self.yaw)*math.cos(self.pitch),math.sin(self.pitch),
                         -math.cos(self.yaw)*math.cos(self.pitch)],np.float32)

    @property
    def target(self):return self.eye+self.forward

    def look(self,dx,dy):
        self.yaw+=dx*.003
        self.pitch=float(np.clip(self.pitch-dy*.003,-1.54,1.54))

    def scroll(self,amount):self.speed=float(np.clip(self.speed*math.exp(amount*.12),.01,10000))

    def move(self,keys,dt):
        forward=self.forward;right=normalize(np.cross(forward,[0,1,0]))
        direction=(('w' in keys)-('s' in keys))*forward+(('d' in keys)-('a' in keys))*right
        direction[1]+=('e' in keys)-('q' in keys)
        if np.linalg.norm(direction)>0:
            boost=4 if {'left_shift','right_shift'}&keys else 1
            self.eye+=normalize(direction)*self.speed*boost*dt


class DayClock:
    def __init__(self,hour=10.,minutes=20.):
        self.hour=hour%24;self.minutes=minutes;self.running=True
        self.remaining=0.;self.elapsed=0.;self.duration=.8

    def set(self,hour):self.hour=hour%24;self.remaining=0.;self.elapsed=0.

    def nudge(self,hours):self.remaining+=hours;self.elapsed=0.

    def tick(self,dt):
        dt=max(dt,0.)
        if self.running:self.hour+=dt*24/(max(self.minutes,1)*60)
        if self.remaining:
            old=self.elapsed/self.duration
            self.elapsed=min(self.duration,self.elapsed+dt);new=self.elapsed/self.duration
            ease=lambda x:x*x*(3-2*x)
            fraction=(ease(new)-ease(old))/max(1-ease(old),1e-8)
            delta=self.remaining*fraction;self.remaining-=delta;self.hour+=delta
            if new>=1:self.remaining=0.
        self.hour%=24
        return self.hour
