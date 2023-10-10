import time
from numpy import rot90
#from cmath import cos,sin
from math import cos,sin
import numpy as np

import torch

from meshplot.Viewer import Viewer as ori_Viewer


class Viewer(ori_Viewer):
    def __init__(self, settings):
        super(Viewer,self).__init__(settings)
        # self._cam = p3s.PerspectiveCamera(position=[0, 0, 1], lookAt=[0, 0, 0], fov=self.__s["fov"],
        #                         aspect=self.__s["width"]/self.__s["height"], children=[self._light])
        #self._cam = p3s.OrthographicCamera(position=[0, 0, 1], lookAt=[0, 0, 0], children=[self._light])

        #print(type(self._cam ))

    
    # 注意，这不是源码
    def update_points(self, oid=0, vertices=None):
        #print(type(self._cam ))
        #print(len(self.__objects))
        obj = self.__objects[oid]
        if type(vertices) != type(None):
            
            v = vertices.astype("float32", copy=False)
            
            obj["geometry"].attributes["position"].array = v
            #self.wireframe.attributes["position"].array = v # Wireframe updates?
            obj["geometry"].attributes["position"].needsUpdate = True
            
            # print("gg")
            # print(obj["geometry"].attributes["index"].array) 

        from meshplot.plot import rendertype
        if rendertype == "WEBSITE":
            return self
        
        # 注意，这不是源码
    def update_edges(self, oid=0,vertices=None, edges=None):
        obj = self.__objects[oid]
        if (type(edges) != type(None)) and ( type(vertices) != type(None)):
            
            #v = vertices.astype("float32", copy=False)
            #edges = edges.astype("uint32", copy=False).ravel()
            
            
            #v = vertices.astype("float32", copy=False)
            
            # obj["geometry"].attributes["position"].array = v
            # #self.wireframe.attributes["position"].array = v # Wireframe updates?
            # obj["geometry"].attributes["position"].needsUpdate = True
            
                
            if vertices.shape[1] == 2:
                vertices = np.append(
                    vertices, np.zeros([vertices.shape[0], 1]), 1)
            #sh = self.__get_shading(shading)
            lines = np.zeros((edges.size, 3))
            cnt = 0
            for e in edges:
                lines[cnt, :] = vertices[e[0]]
                lines[cnt+1, :] = vertices[e[1]]
                cnt += 2
                
            lines=lines.astype("float32", copy=False)

            obj["geometry"].attributes["position"].array = lines
            #self.wireframe.attributes["position"].array = v # Wireframe updates?
            obj["geometry"].attributes["position"].needsUpdate = True

            # print(lines.shape)
            
            # print("geometry")
            # #print(obj)
            # print(obj["geometry"]) 
            # print(obj["geometry"].attributes["position"])  
            # print("mesh:")
            # #print(obj)
            # print(obj["mesh"])
            # print(obj["mesh"].geometry.attributes["position"])


            # obj["mesh"].attributes["index"].array = edges
            # #self.wireframe.attributes["position"].array = v # Wireframe updates?
            # obj["mesh"].attributes["index"].needsUpdate = True
            
            

        from meshplot.plot import rendertype
        if rendertype == "WEBSITE":
            return self


rendertype = "JUPYTER" # "OFFLINE"
def jupyter():
    global rendertype
    rendertype = "JUPYTER"

def offline():
    global rendertype
    rendertype = "OFFLINE"

def website():
    global rendertype
    rendertype = "WEBSITE"

from IPython.display import display
def m_plot(v, f=None, c=None, uv=None, n=None, shading={}, plot=None, return_plot=True, filename="", texture_data=None):#, return_id=False):
    if not plot:
        view = Viewer(shading)
    else:
        view = plot
        view.reset()
    if type(f) == type(None): # Plot pointcloud
        obj_id = view.add_points(v, c, shading=shading)
    elif type(f) == np.ndarray and len(f.shape) == 2 and f.shape[1] == 2: # Plot edges
        obj_id = view.add_edges(v, f, shading=shading)
    else: # Plot mesh
        obj_id = view.add_mesh(v, f, c, uv=uv, n=n, shading=shading, texture_data=texture_data)

    if not plot and rendertype == "JUPYTER":
        display(view._renderer)

    if rendertype == "OFFLINE":
        view.save(filename)

    if return_plot or rendertype == "WEBSITE":
        return view

    
    
SMPL_JOINT = torch.tensor([[23, 21], [21, 19], [19, 17], [17, 14], [14, 9], 
            [9, 13], [13, 16], [16, 18], [18, 20], [20, 22], 
            [15, 12], [12, 9],[9, 6], [6, 3], [3, 0], 
            [0, 2], [2, 5],[5, 8], [8, 11], 
            [0, 1], [1, 4], [4, 7],[7, 4], [7, 10]], dtype=np.int)

CMU_JOINT = torch.tensor([[0, 1], [1, 2], [2, 3], 
                          [0, 12], [12, 13], [13, 14], [14, 15], 
                          [0, 16], [16, 17], [17, 18], [18, 19], 
                          [1, 4],[4, 5], [5, 6], [6, 7], 
                          [1, 8], [8, 9],[9, 10], [10, 11]], dtype=np.int)

    
def meshplot_visuals_n_joint_seq_color(list_point_cloud_seq,list_color,joint_type="smpl",save_path=None):
    '''
    list_point_cloud_seq:
        type(list(tensor))
        tensor:shape(n_frames,24,3)
        
    '''
    import ipywidgets as widgets
    from IPython.display import display
    
    n_frames,n_vertices,_=list_point_cloud_seq[0].shape
    
    inter_distance=1.0
    
    # 帧间隔时间
    time_interval= 100


    # joint_edge = np.array([[23, 21], [21, 19], [19, 17], [17, 14], [14, 9], 
    #                 [9, 13], [13, 16], [16, 18], [18, 20], [20, 22], 
    #                 [15, 12], [12, 9],[9, 6], [6, 3], [3, 0], 
    #                 [0, 2], [2, 5],[5, 8], [8, 11], 
    #                 [0, 1], [1, 4], [4, 7],[7, 4], [7, 10]], dtype=np.int)
    if joint_type=="smpl":
        joint_edge=SMPL_JOINT
    elif joint_type=="cmu":
        joint_edge=CMU_JOINT

    
    #joint_edge_2=joint_edge_zero+24*1

    #joint_edge=joint_edge_zero
    point_size=0.3

    pc=list_point_cloud_seq[0][0].cpu().detach().numpy()
    joint_edge=joint_edge.cpu().detach().numpy() 
    print(pc.shape)
    
    viewer=m_plot(pc,shading={"point_size":point_size,"point_color": list_color[0],"width": 2000, "height": 500})   
    viewer.add_edges(pc,joint_edge,shading={"line_color": list_color[0]})
    
    for i in range(0,len(list_point_cloud_seq)) :
        
        if i>0:
            temp_data=list_point_cloud_seq[i][0]+torch.Tensor([inter_distance*i,0,0]).cuda()
            temp_data=temp_data.cpu().detach().numpy()
            viewer.add_points(temp_data,shading={ "point_size": point_size, "point_color": list_color[i]})
            viewer.add_edges(temp_data,joint_edge,shading={"line_color": list_color[i]})

    #viewer._check()
    
    play = widgets.Play(
    value=0,
    min=0,
    max=n_frames-1,
    step=1,
    interval=time_interval,
    description="Press play",
    disabled=False
    )
    slider = widgets.IntSlider(
        min=0,
        max=n_frames-1
    )
    widgets.jslink((play, 'value'), (slider, 'value'))
    ui=widgets.HBox([play, slider])


    #print(play.value)

    def event(play):
        for i in range(0,len(list_point_cloud_seq)) :
            #i=1
            temp_data=list_point_cloud_seq[i][play]+torch.Tensor([inter_distance*i,0,0]).cuda()
            temp_data=temp_data.cpu().detach().numpy()
            #temp_data=temp_data.cpu().detach().numpy()
            # viewer.add_points(temp_data,shading={ "point_size": 0.15, "point_color": list_color[i]})
            # viewer.add_edges(temp_data,joint_edge,shading={"line_color": list_color[i]})

            #pc=temp_data[play].cpu().detach().numpy()
        
            viewer.update_points(oid=2*i,vertices=temp_data)
            viewer.update_edges(oid=2*i+1,vertices=temp_data, edges=joint_edge)

        
    out=widgets.interactive_output(
        event,
        {
            'play':play
        }
    )
    display(ui,out)   
    
    if save_path!=None:   
        viewer.save(save_path)

def meshplot_visuals_n_seq_color(list_point_cloud_seq,list_color,save_path=None):
    '''
    list_point_cloud_seq:
        type(list(tensor))
        tensor:shape(n_frames,24,3)
        
    '''
    import ipywidgets as widgets
    from IPython.display import display
    
    n_frames,n_vertices,_=list_point_cloud_seq[0].shape
    point_size=0.2
    
    inter_distance=1.0
    
    # 帧间隔时间
    time_interval= 100


    pc=list_point_cloud_seq[0][0].cpu().detach().numpy()
    #print(pc.shape)
    
    if isinstance(list_color[0],str):
        viewer=m_plot(pc,shading={"point_size": point_size,"point_color": list_color[0],"width": 1200, "height": 800})  
    else: 
        viewer=m_plot(v=pc,c=list_color[0],shading={"point_size": point_size,"width": 1200, "height": 800,"colormap":'viridis'})
        #viewer.add_points(pc[2:3,:],shading={ "point_size": point_size, "point_color": 'red'})  

    for i in range(0,len(list_point_cloud_seq)) :
        
        if i>0:
            temp_data=list_point_cloud_seq[i][0]+torch.Tensor([inter_distance*i,0,0]).cuda()
            temp_data=temp_data.cpu().detach().numpy()
            if isinstance(list_color[0],str):
                viewer.add_points(temp_data,shading={ "point_size": point_size, "point_color": list_color[i]})
                
            else:
                viewer.add_points(temp_data,c=list_color[i],shading={ "point_size": point_size})
                #viewer.add_points(temp_data[2:3,:],shading={ "point_size": point_size, "point_color": 'red'})


    #viewer._check()
    
    
    play = widgets.Play(
    value=0,
    min=0,
    max=n_frames-1,
    step=1,
    interval=time_interval,
    description="Press play",
    disabled=False
    )
    slider = widgets.IntSlider(
        min=0,
        max=n_frames-1
    )
    widgets.jslink((play, 'value'), (slider, 'value'))
    ui=widgets.HBox([play, slider])


    #print(play.value)

    def event(play):
        for i in range(0,len(list_point_cloud_seq)) :
            #i=1
            temp_data=list_point_cloud_seq[i][play]+torch.Tensor([inter_distance*i,0,0]).cuda()
            temp_data=temp_data.cpu().detach().numpy()
            #temp_data=temp_data.cpu().detach().numpy()
            # viewer.add_points(temp_data,shading={ "point_size": 0.15, "point_color": list_color[i]})
            # viewer.add_edges(temp_data,joint_edge,shading={"line_color": list_color[i]})

            #pc=temp_data[play].cpu().detach().numpy()
        
            viewer.update_points(oid=i,vertices=temp_data)
            #viewer.update_points(oid=i+len(list_point_cloud_seq),vertices=temp_data[2:3,:])


        
    out=widgets.interactive_output(
        event,
        {
            'play':play
        }
    )
    display(ui,out)
    if save_path!=None:   
        viewer.save(save_path)




def meshplot_visuals_n_seq_mesh_color(list_point_cloud_seq,list_face_seq,list_color,save_path=None):
    '''
    list_point_cloud_seq:
        type(list(tensor))
        tensor:shape(n_frames,24,3)
        
    '''
    import ipywidgets as widgets
    from IPython.display import display
    
    n_frames,n_vertices,_=list_point_cloud_seq[0].shape

    point_size=0.2
    
    inter_distance=1.0
    
    # 帧间隔时间
    time_interval= 100


    pc=list_point_cloud_seq[0][0].cpu().detach().numpy()
    faces=list_face_seq[0][0].cpu().detach().numpy()

    color=pc[:, 1]
    print(color.size)
    #print(faces)
    
    if isinstance(list_color[0],str):
        viewer=m_plot(v=pc,f=faces,c=color,shading={"width": 2000, "height": 800})  
    else: 
        viewer=m_plot(v=pc,f=faces,c=color,shading={"width": 2000, "height": 800,"colormap":'viridis'})
        #viewer.add_points(pc[2:3,:],shading={ "point_size": point_size, "point_color": 'red'})  

    for i in range(0,len(list_point_cloud_seq)) :
        
        if i>0:
            temp_data=list_point_cloud_seq[i][0]+torch.Tensor([inter_distance*i,0,0]).cuda()
            temp_data=temp_data.cpu().detach().numpy()
            if isinstance(list_color[0],str):
                viewer.add_mesh(temp_data,f=faces,c=color)
                
            else:
                viewer.add_mesh(temp_data,f=faces,c=color)
                #viewer.add_points(temp_data[2:3,:],shading={ "point_size": point_size, "point_color": 'red'})


    #viewer._check()
    
    
    play = widgets.Play(
    value=0,
    min=0,
    max=n_frames-1,
    step=1,
    interval=time_interval,
    description="Press play",
    disabled=False
    )
    slider = widgets.IntSlider(
        min=0,
        max=n_frames-1
    )
    widgets.jslink((play, 'value'), (slider, 'value'))
    ui=widgets.HBox([play, slider])


    #print(play.value)

    def event(play):
        for i in range(0,len(list_point_cloud_seq)) :
            #i=1
            temp_data=list_point_cloud_seq[i][play]+torch.Tensor([inter_distance*i,0,0]).cuda()
            temp_data=temp_data.cpu().detach().numpy()
            #temp_data=temp_data.cpu().detach().numpy()
            # viewer.add_points(temp_data,shading={ "point_size": 0.15, "point_color": list_color[i]})
            # viewer.add_edges(temp_data,joint_edge,shading={"line_color": list_color[i]})

            #pc=temp_data[play].cpu().detach().numpy()
        
            viewer.update_object(oid=i,vertices=temp_data)
            #viewer.update_points(oid=i+len(list_point_cloud_seq),vertices=temp_data[2:3,:])


        
    out=widgets.interactive_output(
        event,
        {
            'play':play
        }
    )
    display(ui,out)
    if save_path!=None:   
        viewer.save(save_path)



def meshplot_visuals_n_seq_pointcloudandmesh_color(list_point_cloud_seq,list_surface_points_seq,list_face_seq,list_color,save_path=None):
    '''
    list_point_cloud_seq:
        type(list(tensor))
        tensor:shape(n_frames,24,3)
        
    '''
    import ipywidgets as widgets
    from IPython.display import display
    
    n_frames,n_vertices,_=list_surface_points_seq[0].shape

    point_size=0.2
    
    inter_distance=1.0

    point_offset=0
    
    # 帧间隔时间
    time_interval= 100


    surface_pc=list_surface_points_seq[0][0].cpu().detach().numpy()
    faces=list_face_seq[0][0].cpu().detach().numpy()

    pc=list_point_cloud_seq[0][0]
    color=surface_pc[:, 1]
    pc=pc+torch.Tensor([point_offset,0,0]).cuda()
    
    pc=pc.cpu().detach().numpy()
    #print(faces)
    
    if isinstance(list_color[0],str):
        viewer=m_plot(v=surface_pc,f=faces,c=color,shading={"width": 1000, "height": 800}) 
        viewer.add_points(pc,shading={ "point_size": point_size, "point_color": 'red'})  
    else: 
        viewer=m_plot(v=surface_pc,f=faces,c=color,shading={"width": 1000, "height": 800,"colormap":'viridis'})
        viewer.add_points(pc,shading={ "point_size": point_size, "point_color": 'red'}) 
        #viewer.add_points(pc[2:3,:],shading={ "point_size": point_size, "point_color": 'red'})  

    for i in range(0,len(list_surface_points_seq)) :
        
        if i>0:
            temp_data=list_surface_points_seq[i][0]+torch.Tensor([inter_distance*i,0,0]).cuda()
            temp_data=temp_data.cpu().detach().numpy()
            temp_pc=list_point_cloud_seq[i][0]+torch.Tensor([inter_distance*i,0,0]).cuda()
            temp_pc=temp_pc.cpu().detach().numpy()
            if isinstance(list_color[0],str):
                viewer.add_mesh(temp_data,f=faces)
            else:
                viewer.add_mesh(temp_data,f=faces,c=color,shading={"colormap":'viridis'})
                viewer.add_points(temp_pc,shading={ "point_size": point_size, "point_color": 'red'})


    #viewer._check()
    
    
    play = widgets.Play(
    value=0,
    min=0,
    max=n_frames-1,
    step=1,
    interval=time_interval,
    description="Press play",
    disabled=False
    )
    slider = widgets.IntSlider(
        min=0,
        max=n_frames-1
    )
    widgets.jslink((play, 'value'), (slider, 'value'))
    ui=widgets.HBox([play, slider])


    #print(play.value)

    #print(list_surface_points_seq[0].shape)
    def event(play):
        for i in range(0,len(list_surface_points_seq)) :
            #i=1
            temp_data=list_surface_points_seq[i][play]+torch.Tensor([inter_distance*i,0,0]).cuda()
            temp_data=temp_data.cpu().detach().numpy()
            temp_point_cloud=list_point_cloud_seq[i][play]+torch.Tensor([inter_distance*i,0,0]).cuda()+torch.Tensor([point_offset,0,0]).cuda()

            temp_point_cloud=temp_point_cloud.cpu().detach().numpy()


            #temp_data=temp_data.cpu().detach().numpy()
            # viewer.add_points(temp_data,shading={ "point_size": 0.15, "point_color": list_color[i]})
            # viewer.add_edges(temp_data,joint_edge,shading={"line_color": list_color[i]})

            #pc=temp_data[play].cpu().detach().numpy()
            if len(list_surface_points_seq)==1:
                viewer.update_object(oid=i,colors=color,vertices=temp_data)
                viewer.update_points(oid=i+len(list_point_cloud_seq),vertices=temp_point_cloud)


        
    out=widgets.interactive_output(
        event,
        {
            'play':play
        }
    )
    display(ui,out)
    if save_path!=None:   
        viewer.save(save_path)