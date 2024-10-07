from typing import Union, Optional

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio

from pytorch3d.structures import Meshes
import open3d as o3d
from source.utils.data_utils import to_numpy
from source.utils.decor_utils import auto_numpy
from source.visualization.pca_colormap import apply_pca_colormap
from source.visualization.plot_utils import update_scene_layout, _pointcloud_trace, \
    _volume_trace, _flow_traces_v1
from matplotlib import colormaps as cmaps

pio.renderers.default = "browser"


@auto_numpy
def plot_pointcloud(data: np.ndarray,
                    fig: Optional[go.Figure] = None,
                    downsample_ratio: Optional[int] = 1,
                    pts_size: float = 1.8,
                    colors: Optional[Union[str, list, np.ndarray]] = None,
                    colorscale: str = 'inferno',  # viridis spectral RdYlGn
                    order_as_color: Optional[bool] = False,
                    name=None,
                    draw_line=False,
                    row=None,
                    col=None,
                    scene_id=1,
                    # layout parameters
                    update_layout=True,
                    center=True, view='top', dis=2, show_grid=True,
                    margin=0.1, height=400, width=None,
                    **kwargs):
    """
    Plot a point cloud using plotly.

    Args:
        data: point cloud data, shape (N, 3).
        fig: plotly figure object.
        downsample_ratio: downsample ratio for plotting.
        pts_size: size of points.
        colors: color of each point.
            can be a string, a list of colors, a numpy array of shape (N, 3), or
            a numpy array of shape (N,) which will get mapped to rgb using color scale.
        colorscale: color scale for mapping scalar to color.
        order_as_color: whether to use order of points as color
        row: row index of subplot.
        col: column index of subplot.
        scene_id: scene id.


    """
    if fig is None:
        fig = go.Figure()
    if order_as_color:
        assert colors is None
        colors = np.arange(data.shape[0])
    fig.add_trace(
        _pointcloud_trace(data, downsample_ratio, colors=colors, pts_size=pts_size,
                          colorscale=colorscale, draw_line=draw_line, name=name,
                          **kwargs),
        row=row, col=col)
    if update_layout:
        update_scene_layout(
            fig, scene_id=scene_id, pts=data, center=center, view=view, dis=dis,
            show_grid=show_grid, margin=margin, height=height, width=width)
    return fig


@auto_numpy
def plot_seg_fig(data: np.ndarray,
                 labels: np.ndarray,
                 colormap=None,
                 labelmap=None, fig=None, show_grid=True, height=400, width=None,
                 pts_size=1.8,
                 **kwargs):
    # Create a figure.
    if fig is None:
        fig = go.Figure()

    # Colormap.
    labels = labels.astype(int)
    for label in np.unique(labels):
        # pdb.set_trace()
        subset = data[np.where(labels == label)]
        if colormap is not None:
            color = colormap[label]
            colors = [color] * subset.shape[0]
        else:
            colors = None
        # subset = np.squeeze(subset)
        if labelmap is not None:
            legend = labelmap[label]
        else:
            legend = str(label)
        fig.add_trace(
            _pointcloud_trace(subset, colors=colors, name=legend, pts_size=pts_size))
    fig.update_layout(showlegend=True)

    update_scene_layout(fig, pts=data, show_grid=show_grid, height=height, width=width,
                        **kwargs)

    return fig


@auto_numpy
def plot_pointclouds(pcs,
                     colors=None, labels=None, fig=None, show_grid=True,
                     order_as_color=False,
                     labelmap=None, **kwargs):
    """Plot multiple point clouds at a time

    Args:
        pcs: list of point clouds, each of shape (N, 3) or ndarray of (B, N, 3)
        colors: list of colors for each point cloud
        labels: list of labels for each point cloud
        fig: plotly figure object
        show_grid: whether to show grid
        labelmap: mapping from label to legend
        **kwargs: other parameters for plot_pointcloud
    """
    num_pc = len(pcs)
    color = None
    name = None
    for i in range(num_pc):
        if colors is not None:
            color = colors[i]
        # TODO: add support of label map
        if labels is not None:
            name = labels[i]
        fig = plot_pointcloud(pcs[i], fig=fig, colors=color, name=name,
                              order_as_color=order_as_color,
                              update_layout=False, show_grid=show_grid, **kwargs)

    # update the layout using all the points
    if isinstance(pcs, np.ndarray):
        pcs = pcs.reshape(-1, 3)
    elif isinstance(pcs, list):
        pcs = np.concatenate(pcs)
    update_scene_layout(fig, pts=pcs, show_grid=show_grid)

    return fig


def plot_latent_pointcloud(pc, pc_feat, project_info=None, pts_size=3, **kwargs):
    pc = pc.reshape(-1, 3)
    pc_feat = pc_feat.reshape(-1, pc_feat.shape[-1])
    with torch.autocast("cuda", dtype=torch.float32):
        pca_color, project_info = apply_pca_colormap(pc_feat.float(),
                                                     project_info=project_info)
    pca_color = pca_color * 255
    pca_color = torch.clamp(pca_color.int(), 0, 255)
    fig = plot_pointcloud(pc, colors=pca_color, pts_size=pts_size, **kwargs)
    return fig, project_info


def plot_latent_pointclouds(pcs, pc_feats, fig=None, project_info=None, **kwargs):
    if fig is None:
        fig = go.Figure()
    for pc, pc_feat in zip(pcs, pc_feats):
        pc = pc.reshape(-1, 3)
        pc_feat = pc_feat.reshape(-1, pc_feat.shape[-1])
        fig, project_info = plot_latent_pointcloud(pc, pc_feat,
                                                   project_info=project_info, fig=fig,
                                                   **kwargs)
    return fig

def plot_latent_mesh(
        v=None, f=None,
        mesh: Optional[Union[Meshes, o3d.geometry.TriangleMesh]] = None,
        v_feat=None, fig=None, project_info=None,
        opacity=0.6,
        **kwargs
):
    v_feat = v_feat.reshape(-1, v_feat.shape[-1])
    with torch.autocast("cuda", dtype=torch.float32):
        pca_color, project_info = apply_pca_colormap(v_feat.float(),
                                                     project_info=project_info)
    pca_color = pca_color * 255
    pca_color = torch.clamp(pca_color.int(), 0, 255)
    # pca_color = None
    fig = plot_mesh(v=v, f=f, mesh=mesh, fig=fig, vertexcolor=pca_color, opacity=opacity,
                    **kwargs)
    return fig, project_info


'''
Plot 3D mesh, which is a combination of Mesh3d and Scatter3d.
'''


@auto_numpy
def plot_mesh(v=None, f=None, verts_normals=None,
              mesh: Optional[Union[Meshes, o3d.geometry.TriangleMesh]] = None,
              fig=None,
              view='top', row=None, col=None, id=1, dis=2, opacity=0.6,
              draw_face=True,
              show_grid=True,
              vertexcolor=None,
              **kwargs):
    """Plot Meshes of pytorch3d or open3d
    """

    if fig is None:
        fig = go.Figure()
    if mesh is not None:
        if isinstance(mesh, Meshes):
            v = to_numpy(mesh.verts_packed())
            f = to_numpy(mesh.faces_packed())
        elif isinstance(mesh, o3d.geometry.TriangleMesh):
            v = np.asarray(mesh.vertices)
            f = np.asarray(mesh.triangles)
    v = v.reshape(-1, 3)
    f = f.reshape(-1, 3)

    fig = fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                                  i=f[:, 0], j=f[:, 1], k=f[:, 2], # colorscale='RdYlBu',
                                  vertexcolor=vertexcolor,
                                  opacity=opacity),
                        row=row, col=col)
    if draw_face:
        tri_pts = v[f]
        Xe = []
        Ye = []
        Ze = []
        for T in tri_pts:
            Xe.extend([T[k % 3][0] for k in range(4)] + [None])
            Ye.extend([T[k % 3][1] for k in range(4)] + [None])
            Ze.extend([T[k % 3][2] for k in range(4)] + [None])
        fig = fig.add_trace(go.Scatter3d(x=Xe,
                                         y=Ye,
                                         z=Ze,
                                         mode='lines'
                                         ), row=row, col=col)
    if verts_normals is not None:
        fig.add_trace(_flow_traces_v1(
            v, verts_normals, sizeref=100
        ))
        # fig.add_trace(_flow_traces_v2(
        #     v, verts_normals, sizeref=0.03
        # ))
    update_scene_layout(fig, scene_id=id, pts=v, view=view, show_grid=show_grid,
                        dis=dis, **kwargs)
    return fig


@auto_numpy
def plot_cone(v, flow, fig=None,
              view='top', row=None, col=None, id=1, dis=2,
              show_grid=True, sizeref=1, update_layout=True, **kwargs):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(_flow_traces_v1(v, flow, sizeref=sizeref), row=row, col=col)
    if update_layout:
        update_scene_layout(fig, scene_id=id, pts=v, view=view, show_grid=show_grid,
                            dis=dis, **kwargs)

    return fig


@auto_numpy
def plot_arrow(pos, flow, fig=None, view='top', color='red',name="flow", row=None, col=None, id=1, dis=2,
               show_grid=True, sizeref=1., update_layout=True, **kwargs):
    if fig is None:
        fig = go.Figure()
    x_lines = list()
    y_lines = list()
    z_lines = list()

    # normalize flows:
    nonzero_flows = np.all(flow == 0.0, axis=-1)
    n_pos = pos[~nonzero_flows]
    n_flows = flow[~nonzero_flows]
    n_dest = n_pos + n_flows * sizeref
    if n_pos.shape[0] == 0:
        return fig
    for i in range(len(n_pos)):
        x_lines.append(n_pos[i][0])
        y_lines.append(n_pos[i][1])
        z_lines.append(n_pos[i][2])
        x_lines.append(n_dest[i][0])
        y_lines.append(n_dest[i][1])
        z_lines.append(n_dest[i][2])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
        
    lines_trace = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        line=dict(color=color, width=5),
        name=name,
        opacity=0.7
    )
    
    fig.add_trace(lines_trace)
    
    cone_flow = n_flows * 0.3 * sizeref
    flow_trace = go.Cone(
        x=n_dest[:, 0],
        y=n_dest[:, 1],
        z=n_dest[:, 2],
        u=cone_flow[:, 0],
        v=cone_flow[:, 1],
        w=cone_flow[:, 2],
        cmax=1,
        cmin=-10000,
        colorscale=[[0, color], [1,color]],
        sizemode="raw",
        sizeref=1,
        anchor="tip",
        showscale=False
    )
    fig.add_trace(flow_trace)
    if update_layout:
        pts = np.concatenate([n_pos, n_dest], axis=0)
        update_scene_layout(fig, scene_id=id, pts=pts, view=view, show_grid=show_grid,
                            dis=dis, **kwargs)

    return fig

@auto_numpy
def plot_volume(volume, fig=None, view='top', dis=2, show_grid=True, id=1, row=None,
                col=None):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(_volume_trace(volume), row=row, col=col)
    update_scene_layout(fig, scene_id=id, view=view, dis=dis, show_grid=show_grid)
    return fig


@auto_numpy
def plot_picker_traj(pick_pos=None, actions=None, picker_traj=None, plot_dim=[2, 1],
                     save_path=None, is_softgym=False, rewards=None, table_cfg=None):
    if picker_traj is None:
        picker_traj = pick_pos.reshape(1, 3) + np.cumsum(actions[:, :3],
                                                         axis=0)  # (H, 3)
        picker_traj = np.concatenate([pick_pos.reshape(1, 3), picker_traj], axis=0)
        # print('init and end ', picker_trajs[0], picker_trajs[18:22])
    if len(plot_dim) == 2:
        if len(picker_traj.shape) == 2:
            picker_traj = picker_traj[None]
        if is_softgym:
            picker_traj[..., 0] = -picker_traj[..., 0]
        picker_traj = picker_traj[..., plot_dim]
        if rewards is not None:
            rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())
            cmap = cmaps['viridis']
            colors = cmap(rewards)
        else:
            colors = [None] * picker_traj.shape[0]
        fig, ax = plt.subplots()
        if table_cfg is not None:
            trans_x, h = table_cfg[:2]
            ax.add_patch(plt.Rectangle((trans_x - 0.4, 0), 0.8, h))
            obs_cfgs = table_cfg[2:].reshape(-1, 6)
            num_obs = len(obs_cfgs)
            for i in range(num_obs):
                obs_cfg = obs_cfgs[i]
                ax.add_patch(
                    plt.Rectangle((obs_cfg[0] - obs_cfg[3], obs_cfg[2] - obs_cfg[5]),
                                  obs_cfg[3] * 2, obs_cfg[5] * 2))
        for i in range(picker_traj.shape[0]):
            # plt.plot(picker_traj[i, :, 0], picker_traj[i, :, 1], color=colors[i])
            plt.plot(picker_traj[i, :, 0], picker_traj[i, :, 1], color="yellow")
        plt.scatter(picker_traj[0, 0, 0], picker_traj[0, 0, 1], color='red')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
    elif len(plot_dim) == 3:
        plot_pointclouds([picker_traj, picker_traj[0:1]]).show()


@auto_numpy
def plot_plan_traj(pick_pos=None, actions=None,
                   prev_nominal=None, cur_nominal=None, plot_dim=[2, 1],
                   save_path=None, is_softgym=False, rewards=None):
    actions = np.concatenate([actions, prev_nominal[None], cur_nominal[None]], axis=0)
    actions = np.concatenate([np.zeros((actions.shape[0], 1, 3)), actions], axis=1)
    picker_trajs = pick_pos.reshape(1, 1, 3) + np.cumsum(actions[:, :, :3],
                                                         axis=1)  # (K, H, 3)
    # picker_trajs = np.concatenate([, picker_trajs], axis=0)
    picker_trajs = picker_trajs[..., plot_dim]
    if rewards is not None:
        rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())
        cmap = cmaps['viridis']
        colors = cmap(rewards)
        colors = [x for x in colors]
    else:
        colors = [None] * picker_trajs.shape[0]
    colors += ['blue', 'red']
    if is_softgym:
        picker_trajs[..., 0] = -picker_trajs[..., 0]
    for i in range(picker_trajs.shape[0]):
        plt.plot(picker_trajs[i, :, 0], picker_trajs[i, :, 1], color=colors[i])
    plt.scatter(picker_trajs[0, 0, 0], picker_trajs[0, 0, 1], color='red')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


@auto_numpy
def plot_hinge_from_key_pts(key_pts, fig=None, show_grid=True, return_pts=False):
    """
    First 2 points: axis
    Next 2 points: page
    """
    if fig is None:
        fig = go.Figure()
    if len(key_pts.shape) == 3:
        key_pts = key_pts[0]
    if key_pts.shape[0] == 4:
        vs = np.zeros((6, 3))
        vs[:2] = key_pts[:2]
        for i, page_id in enumerate([2, 3]):
            vs[2 * i + 2] = 2 * key_pts[page_id] - key_pts[0]
            vs[2 * i + 3] = 2 * key_pts[page_id] - key_pts[1]

        f = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4], [0, 4, 5]])
    elif key_pts.shape[0] == 10:
        vs = key_pts[2:]
        f = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        fig = fig.add_trace(go.Scatter3d(x=key_pts[:2, 0],
                                         y=key_pts[:2, 1],
                                         z=key_pts[:2, 2],
                                         mode='lines',
                                         line=dict(width=20),
                                         ))

    plot_mesh(vs, f, fig=fig, show_grid=show_grid, draw_face=False)
    fig.update_layout(showlegend=True)

    update_scene_layout(fig, pts=vs, show_grid=show_grid)
    if return_pts:
        return fig, vs
    return fig


@auto_numpy
def plot_hinge_from_key_pts_compare(key_pts, pred_key_pts=None, pc=None, fig=None,
                                    show_grid=True,
                                    data_id=0):
    """
    First 2 points: axis
    Next 2 points: page
    """
    if fig is None:
        fig = go.Figure()
    if pc is not None:
        plot_pointcloud(pc, fig=fig, show_grid=show_grid, downsample_ratio=1)
    fig, vs = plot_hinge_from_key_pts(key_pts, fig=fig, show_grid=show_grid,
                                      return_pts=True)
    if pred_key_pts is not None:
        fig, vs2 = plot_hinge_from_key_pts(pred_key_pts, fig=fig, show_grid=show_grid,
                                           return_pts=True)
        vs = np.concatenate([vs, vs2], axis=0)

    update_scene_layout(fig, pts=vs, show_grid=show_grid)
    fig.update_layout(
        title={
            # 'text': f"Flow EPE: {epe:.3f}     Visibility pred acc: {(gt_vis[valid_pred] == pred_vis).mean():.3f}",
            'text': f"Index {data_id}",
            'y': 0.99,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        }
    )
    return fig


"""
histogram plot 
"""


def hists_plot(xs, labels=None, **kwargs):
    df = pd.DataFrame(dict(
        series=np.concatenate([[l] * len(x) for x, l in zip(xs, labels)]),
        data=np.concatenate(xs)
    ))
    f = px.histogram(df, x="data", color="series", barmode="overlay",
                     histnorm="percent",
                     **kwargs)
    return f


@auto_numpy
def plot_confusion_mat(conf_mat, labels):
    """Plot a confusion matrix using plotly"""
    conf_mat = conf_mat[::-1]
    fig = ff.create_annotated_heatmap(conf_mat, x=labels, y=labels[::-1],
                                      colorscale='RdBu')
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 25
    # add custom xaxis title
    fig.add_annotation(
        dict(font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
             x=0.5,
             y=1.05,
             showarrow=False,
             xanchor='center',
             yanchor='top',
             text="Prediction",
             xref="paper",
             yref="paper"))

    # add custom yaxis title
    fig.add_annotation(
        dict(font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
             x=-0.05,
             y=0.5,
             showarrow=False,
             text="Ground truth",
             textangle=-90,
             xref="paper",
             yref="paper"))
    fig.update_layout(margin=dict(t=200, l=200))
    fig['data'][0]['showscale'] = True
    return fig


def _flow_traces_v2(
        pos, flows, sizeref=0.05, scene="scene", flowcolor="red", name="flow"
):
    x_lines = list()
    y_lines = list()
    z_lines = list()

    # normalize flows:
    nonzero_flows = np.all(flows == 0.0, axis=-1)
    n_pos = pos[~nonzero_flows]
    n_flows = flows[~nonzero_flows]
    n_dest = n_pos + n_flows * sizeref
    for i in range(len(n_pos)):
        x_lines.append(n_pos[i][0])
        y_lines.append(n_pos[i][1])
        z_lines.append(n_pos[i][2])
        x_lines.append(n_dest[i][0])
        y_lines.append(n_dest[i][1])
        z_lines.append(n_dest[i][2])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
    lines_trace = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        scene=scene,
        line=dict(color=flowcolor, width=5),
        name=name,
        opacity=0.7
    )

    return lines_trace
