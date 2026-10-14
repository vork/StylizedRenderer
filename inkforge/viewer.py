"""Resizable native GLFW/Slang window, captured fly controls and art direction."""
import textwrap
import time
from pathlib import Path
import slangpy as spy
from .renderer import default_camera
from .environment import LIGHTING_MODES,GROUND_MODES
from .viewer_state import load_dropped_scene,replace_renderer
from .navigation import FlyCamera,DayClock
from .performance import FrameStats


class Viewer:
    def __init__(self,renderer,eye,target,style,fov,raw_scene=None,ground_mode='auto',time_seconds=0.):
        self.renderer=renderer;self.style=style;self.fov=fov
        self.raw_scene=raw_scene or renderer.scene;self.ground_mode=ground_mode
        self.camera=FlyCamera(eye,target);self.clock=DayClock(style.time_of_day)
        self.cloud_time=float(time_seconds);self.animate_clouds=True;self.message='Drop GLB / USD scenes into this window'
        self.pending_files=None;self.pending_ground=None;self.pending_save=False
        self.drag=False;self.last=None;self.keys=set();self.last_time=time.perf_counter();self.debug=0
        self.window=spy.Window(width=renderer.width,height=renderer.height,title='INKFORGE | Black ink / brighter bloom',resizable=True)
        self.surface=renderer.device.create_surface(self.window)
        self.ui=spy.ui.Context(renderer.device)
        self.window.on_keyboard_event=self.on_keyboard_event
        self.window.on_mouse_event=self.on_mouse_event
        self.window.on_drop_files=self.on_drop_files
        self.panel=spy.ui.Window(self.ui.screen,title='INKFORGE / ART DIRECTION',position=spy.float2(16,16),size=spy.float2(490,min(780,renderer.height-32)))
        spy.ui.Text(self.panel,text='Slang / GTAO / GT7 SDR\nRight mouse: look | WASD: fly | Q/E: down/up\nShift: faster | Wheel: speed | F1: settings\nF2: performance | T / Shift+T: time +/-2h\nU: pause day | C: pause clouds\nP: save | Esc: release/quit')
        world=spy.ui.Group(self.panel,label='Sky, time and scene')
        self.time_control=self.slider(world,'Time of day (hours)',style.time_of_day,0,24,self.set_hour,'time')
        self.day_control=spy.ui.CheckBox(world,label='Day/night cycle',value=True,callback=lambda v:setattr(self.clock,'running',bool(v)))
        self.slider(world,'Minutes per day',20.,1,120,lambda v:setattr(self.clock,'minutes',float(v)),'daylength')
        self.lighting_control=self.combo(world,'Lighting',['Auto','Replace all','Add sky + sun','Sky only / keep lights'],LIGHTING_MODES.index(style.lighting_mode),lambda v:setattr(style,'lighting_mode',LIGHTING_MODES[v]),'lighting')
        self.ground_control=self.combo(world,'Ground',['Auto for objects','On','Off'],GROUND_MODES.index(ground_mode),lambda v:setattr(self,'pending_ground',GROUND_MODES[v]),'ground')
        self.cloud_control=spy.ui.CheckBox(world,label='Moving clouds',value=True,callback=lambda v:setattr(self,'animate_clouds',bool(v)))
        self.style_sliders(world,[('cloud_shadows','Cloud shadow strength',0,1),('ambient','Sky fill',0,3),('sun','Sun intensity',0,8)])
        ink=spy.ui.Group(self.panel,label='Ink and shadows')
        self.style_sliders(ink,[('amount','Comic / PBR',0,1),('outline','Ink width',0,4),('outline_wobble','Stroke variation',0,1),
            ('shadow_fill','Shadow fill',0,1),('penumbra_strength','Tiny edge stippling',0,1),('hatching','AO cross-hatching',0,1),
            ('hatch_density','Hatch density',.5,8),('halftone','Surface halftone',0,1),('dot_frequency','Surface ink frequency',2,100)])
        self.combo(ink,'Hatch direction',['Surface U','Surface V','Diagonal +45','Diagonal -45'],style.hatch_direction,lambda v:setattr(style,'hatch_direction',v),'direction')
        color=spy.ui.Group(self.panel,label='Light, bloom and color')
        self.style_sliders(color,[('exposure','Exposure',-2,4),('bloom','Bloom accent strength',0,.3),('bloom_threshold','Bloom threshold',.5,8),
            ('bloom_dot_size','Bloom dot spacing (pixels)',2,12),('ao','GTAO strength',0,3),('ao_radius','GTAO radius',.02,5),
            ('specular_size','GGX highlight size',0,.3),('saturation','Saturation',.5,1.8),('palette','Palette simplification',0,1)])
        inspect=spy.ui.Group(self.panel,label='Camera and diagnostics')
        self.speed_control=self.slider(inspect,'Fly speed (metres/second)',self.camera.speed,.01,100,lambda v:setattr(self.camera,'speed',float(v)),'speed')
        self.combo(inspect,'View',['Beauty','GTAO','Normals','Base color','Sun shadow','Halftone','Hatching','Patterns','Cascades','Cast shadow','Terminator','Printed shadow'],0,lambda v:setattr(self,'debug',v),'debug')
        self.combo(inspect,'Anti-aliasing',['TAA','Off'],0 if style.taa else 1,lambda v:setattr(style,'taa',v==0),'aa')
        self.performance_panel=spy.ui.Window(self.ui.screen,title='PERFORMANCE / F2',position=spy.float2(max(16,renderer.width-336),16),size=spy.float2(320,210))
        self.performance_text=spy.ui.Text(self.performance_panel,text='Measuring frames…')
        self.performance_control=spy.ui.CheckBox(inspect,label='Performance overlay (F2)',value=True,callback=self.set_performance_visible)
        self.frame_stats=FrameStats();self.performance_refresh=0.
        spy.ui.Button(self.panel,label='Save clean PNG',callback=lambda:setattr(self,'pending_save',True))
        self.status=spy.ui.Text(self.panel,text='')

    @staticmethod
    def slider(parent,label,value,low,high,callback,key):
        # Labels above controls cannot be clipped by ImGui's default item width.
        spy.ui.Text(parent,text=label)
        return spy.ui.SliderFloat(parent,label='##'+key,value=value,min=low,max=high,callback=callback)

    @staticmethod
    def combo(parent,label,items,value,callback,key):
        spy.ui.Text(parent,text=label)
        return spy.ui.ComboBox(parent,label='##'+key,items=items,value=value,callback=callback)

    def style_sliders(self,parent,items):
        for field,label,low,high in items:
            self.slider(parent,label,getattr(self.style,field),low,high,lambda v,f=field:setattr(self.style,f,float(v)),field)

    def set_hour(self,value):self.clock.set(float(value));self.style.time_of_day=self.clock.hour

    def set_performance_visible(self,value):
        self.performance_panel.visible=bool(value)
        self.performance_control.value=bool(value)

    def update_performance(self,now):
        self.performance_control.value=self.performance_panel.visible
        if now<self.performance_refresh:return
        self.performance_refresh=now+.25
        stats=self.frame_stats.summary()
        if stats is None:return
        r=self.renderer
        aa='Off' if not self.style.taa or self.debug else '8-sample stationary' if r.stationary_frame>=8 else 'Temporal / moving'
        self.performance_text.text=(f'{stats["fps"]:.1f} FPS | {stats["frame_ms"]:.1f} ms/frame\n'
            f'Frame p95: {stats["p95_ms"]:.1f} ms\n'
            f'CPU encode/submit: {stats["encode_ms"]:.1f} ms\n'
            f'Acquire/present: {stats["wait_ms"]:.1f} ms\n'
            f'{r.width} x {r.height} | {r.backend} | VSync on\n'
            f'AA: {aa}\n2-second wall-time window\nGPU timestamps not collected')

    def release_mouse(self):
        self.drag=False;self.last=None;self.keys.clear();self.window.cursor_mode=spy.CursorMode.normal

    def save(self):
        name=Path.home()/'Pictures'/'Inkforge'/f'capture-{time.strftime("%Y%m%d-%H%M%S")}.png'
        self.renderer.save(name,self.camera.eye,self.camera.target,self.style,self.fov,self.debug,time_seconds=self.cloud_time)
        self.message=f'Saved {name}'

    def on_drop_files(self,files):self.pending_files=list(files);self.message='Loading dropped scene…'

    def update(self,dt):
        if self.pending_files is not None:
            files=self.pending_files;self.pending_files=None
            try:
                replacement,raw=load_dropped_scene(self.renderer,files,self.ground_mode)
                self.renderer=replacement;self.raw_scene=raw
                self.camera=FlyCamera(*default_camera(raw));self.release_mouse()
                self.message=f'Loaded {Path(raw.source).name} | {len(raw.lights)} lights'
            except Exception as error:self.message=f'Could not load scene: {error}'
        if self.pending_ground is not None:
            mode=self.pending_ground;self.pending_ground=None
            try:
                self.renderer=replace_renderer(self.renderer,self.raw_scene,mode)
                self.ground_mode=mode;self.message=f'Ground: {mode}'
            except Exception as error:
                self.ground_control.value=GROUND_MODES.index(self.ground_mode)
                self.message=f'Could not change ground: {error}'
        if self.animate_clouds:self.cloud_time+=dt
        self.style.time_of_day=self.clock.tick(dt);self.time_control.value=self.clock.hour
        self.camera.move(self.keys,dt);self.speed_control.value=self.camera.speed
        if self.pending_save:
            self.pending_save=False
            try:self.save()
            except Exception as error:self.message=f'Could not save: {error}'
        wrap=max(20,int((self.panel.size.x-30)/8))
        message=textwrap.fill(self.message,wrap)
        self.status.text=f'{message}\n{self.renderer.scene.triangle_count:,} triangles | {self.renderer.backend}\n{self.renderer.width} x {self.renderer.height} | {self.clock.hour:05.2f}h\nFly speed: {self.camera.speed:.2f} m/s'

    def frame(self):
        frame_start=time.perf_counter()
        self.window.process_events()
        if self.window.should_close():return False
        w,h=self.window.width,self.window.height
        if w<=0 or h<=0:
            self.last_time=time.perf_counter();self.frame_stats=FrameStats();time.sleep(.02);return True
        if self.surface.config is None or (w,h)!=(self.surface.config.width,self.surface.config.height):
            self.renderer.device.wait()
            self.surface.configure(width=w,height=h,format=spy.Format.bgra8_unorm,vsync=True)
            self.renderer.resize(w,h)
            self.panel.size=spy.float2(min(490,max(200,w-32)),min(780,max(180,h-32)))
            self.panel.position=spy.float2(16,16)
            self.performance_panel.position=spy.float2(max(16,w-336),16)
            self.frame_stats=FrameStats()
        now=time.perf_counter();dt=min(now-self.last_time,.1);self.last_time=now
        self.update(dt)
        self.update_performance(now)
        acquire_start=time.perf_counter()
        surface=self.surface.acquire_next_image()
        if surface is None:return True
        acquired=time.perf_counter()
        self.renderer.resize(surface.width,surface.height)
        self.ui.begin_frame(surface.width,surface.height,self.window)
        encoder=self.renderer.device.create_command_encoder()
        self.renderer.render(self.camera.eye,self.camera.target,self.style,self.fov,self.debug,encoder,time_seconds=self.cloud_time)
        encoder.blit(surface,self.renderer.output)
        self.ui.end_frame(surface,encoder)
        self.renderer.device.submit_command_buffer(encoder.finish())
        submitted=time.perf_counter()
        self.surface.present()
        end=time.perf_counter()
        self.frame_stats.add(end-frame_start,submitted-acquired,acquired-acquire_start+end-submitted)
        return True

    def on_keyboard_event(self,event):
        key=event.key.name.lower()
        consumed=self.ui.handle_keyboard_event(event)
        if event.is_key_release():self.keys.discard(key);return
        if not event.is_key_press():return
        if key=='escape':
            if self.drag:self.release_mouse()
            else:self.window.close()
            return
        if key=='f1':self.panel.visible=not self.panel.visible;return
        if key=='f2':self.set_performance_visible(not self.performance_panel.visible);return
        if consumed:return
        self.keys.add(key)
        if key=='p':self.pending_save=True
        if key=='t':self.clock.nudge(-2 if event.has_modifier(spy.KeyModifier.shift) else 2)
        if key=='u':self.clock.running=not self.clock.running;self.day_control.value=self.clock.running
        if key=='l':
            index=(LIGHTING_MODES.index(self.style.lighting_mode)+1)%len(LIGHTING_MODES)
            self.style.lighting_mode=LIGHTING_MODES[index];self.lighting_control.value=index
        if key=='g':
            index=(GROUND_MODES.index(self.ground_mode)+1)%len(GROUND_MODES)
            self.pending_ground=GROUND_MODES[index];self.ground_control.value=index
        if key=='c':self.animate_clouds=not self.animate_clouds;self.cloud_control.value=self.animate_clouds

    def on_mouse_event(self,event):
        consumed=self.ui.handle_mouse_event(event)
        pos=(event.pos.x,event.pos.y)
        if event.is_button_up() and event.button==spy.MouseButton.right:
            self.release_mouse();return
        if consumed and not self.drag:return
        if event.is_button_down() and event.button==spy.MouseButton.right:
            self.drag=True;self.last=None;self.window.cursor_mode=spy.CursorMode.disabled
        if event.is_scroll():self.camera.scroll(event.scroll.y)
        if event.is_move() and self.drag:
            if self.last is not None:self.camera.look(pos[0]-self.last[0],pos[1]-self.last[1])
            self.last=pos

    def close(self):
        self.release_mouse();self.renderer.device.wait();self.surface.unconfigure();self.window.close()


def show(renderer,eye,target,style,fov,raw_scene=None,ground_mode='auto',time_seconds=0.):
    viewer=Viewer(renderer,eye,target,style,fov,raw_scene,ground_mode,time_seconds)
    try:
        while viewer.frame():pass
    finally:viewer.close()
