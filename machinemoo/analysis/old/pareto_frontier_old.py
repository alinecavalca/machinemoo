import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import numpy as np


class ParetoFrontier():
    def __init__(self):
        pass

    def plot_two_objs(moo_objs, loss_dic=None, title=None, equal=None, color='blue'):

        if loss_dic is None:
            loss_dic = [{'key': 'obj1', 'title': 'Objective 1'}, 
                        {'key': 'obj2', 'title': 'Objective 2'}]
        if title is None:
            title = 'Pareto Frontier (2D)'

        keys = [dic['key'] for dic in loss_dic]
        axis_titles = [dic['title'] for dic in loss_dic]
        
        loss = {keys[0]: [], keys[1]: []}

        for solution in moo_objs:
            loss[keys[0]].append(solution[0])
            loss[keys[1]].append(solution[1])

        loss_dt = pd.DataFrame.from_dict(loss).sort_values(by=keys[0])

        plt.figure(figsize=(6,4))
        # sns.set_palette("Set1")
        if color == 'blue':
            paleta = sns.color_palette("Blues", n_colors=10)
            color = paleta[7]

        if equal is not None:
            plt.scatter(loss_dt[keys[0]], loss_dt[keys[1]], label='Pareto Solution')
            plt.scatter([equal.objs[0]], [equal.objs[1]], color='indianred')
        else:
            plt.scatter(loss_dt[keys[~0]], loss_dt[keys[~1]], c='gray')
            plt.scatter(loss_dt[keys[0]], loss_dt[keys[1]], c=color)
        plt.xlabel(axis_titles[0],fontsize=12)
        plt.ylabel(axis_titles[1],fontsize=12)
        plt.xlim([loss_dt[keys[0]].min()-0.001, loss_dt[keys[0]].max()+0.001])
        plt.ylim([loss_dt[keys[1]].min()-0.001, loss_dt[keys[1]].max()+0.001])
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True)

        plt.savefig(f"images/{title}.png", dpi=600, bbox_inches='tight')
        plt.show()

        return loss_dt

    def plot_two_objs_colorido(moo_objs, loss_dic=None, title=None, equal=None):

        if loss_dic is None:
            loss_dic = [{'key': 'obj1', 'title': 'Objective 1'}, 
                        {'key': 'obj2', 'title': 'Objective 2'}]
        if title is None:
            title = 'Pareto Frontier (2D)'

        keys = [dic['key'] for dic in loss_dic]
        axis_titles = [dic['title'] for dic in loss_dic]
        
        loss = {keys[0]: [], keys[1]: []}

        for solution in moo_objs:
            loss[keys[0]].append(solution[0])
            loss[keys[1]].append(solution[1])

        loss_dt = pd.DataFrame.from_dict(loss).sort_values(by=keys[0])

        plt.figure(figsize=(6,4))
        
        scatter = plt.scatter( loss_dt[keys[0]], loss_dt[keys[1]], 
                                c=loss_dt[keys[1]], cmap='viridis', 
                                label='Solution Set', edgecolors='black', 
                                s=60,  # Aumenta o tamanho dos pontos
                                alpha=0.8  # Translucidez para melhorar a visualização
                            )
        if equal is not None:
            plt.scatter([equal.objs[0]], [equal.objs[1]], color='indianred', 
                        s=100, label='Equal Solution')
        plt.xlabel(axis_titles[0],fontsize=14)
        plt.ylabel(axis_titles[1],fontsize=14)
        plt.xlim([loss_dt[keys[0]].min()-0.001, loss_dt[keys[0]].max()+0.001])
        plt.ylim([loss_dt[keys[1]].min()-0.001, loss_dt[keys[1]].max()+0.001])
        plt.title(title, fontsize=16)
        plt.legend()
        plt.grid(True)

        # plt.savefig(f"{title}.png", dpi=600, bbox_inches='tight')
        plt.show()

        return loss_dt
    def plot_three_objs(moo_objs, loss_dic=None, title=None, equal=None):
        if loss_dic is None:
            loss_dic = [{'key': f'obj{i}', 'title': f'Objective {i}'}\
                            for i in range(len(moo_objs[0]))]

        if title is None:
            title = 'Pareto Frontier'

        keys = [dic['key'] for dic in loss_dic]
        axis_titles = [dic['title'] for dic in loss_dic]
        
        loss = {keys[i]: [] for i in range(len(moo_objs[0]))}

        for solution in moo_objs:
            for i in range(len(moo_objs[0])):
                loss[keys[i]].append(solution[i])

        loss_dt = pd.DataFrame.from_dict(loss).sort_values(by=keys[0])
        x=loss_dt[keys[0]]
        y=loss_dt[keys[1]]
        z=loss_dt[keys[2]]

        fig = plt.figure(figsize=(8,6))
        
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(loss_dt[keys[0]], loss_dt[keys[1]], loss_dt[keys[2]], color='cornflowerblue')
        if equal is not None:
            plt.scatter([equal.objs[0]], [equal.objs[1]], color='indianred')
        ax.set_xlabel(axis_titles[0], fontsize=12)
        ax.set_ylabel(axis_titles[1], fontsize=12)
        ax.set_zlabel(axis_titles[2], fontsize=12)
        ax.set_zlim3d([0, max(loss_dt[keys[2]])])
        plt.title(title, fontsize=14)

        plt.savefig(f"images/{title}.png", dpi=600, bbox_inches='tight')
        plt.show()

        return loss_dt
    
    def plot_three_objs_2(moo_objs, loss_dic=None, title=None, equal=None):
        if loss_dic is None:
            loss_dic = [{'key': f'obj{i}', 'title': f'Objective {i}'}\
                            for i in range(len(moo_objs[0]))]

        if title is None:
            title = 'Pareto Frontier'

        keys = [dic['key'] for dic in loss_dic]
        axis_titles = [dic['title'] for dic in loss_dic]
        
        loss = {keys[i]: [] for i in range(len(moo_objs[0]))}

        for solution in moo_objs:
            for i in range(len(moo_objs[0])):
                loss[keys[i]].append(solution[i])

        loss_dt = pd.DataFrame.from_dict(loss).sort_values(by=keys[0])
        x=loss_dt[keys[0]]
        y=loss_dt[keys[1]]
        z=loss_dt[keys[2]]

        fig = plt.figure(figsize=(8,6))
        
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(loss_dt[keys[0]], loss_dt[keys[1]], loss_dt[keys[2]], color='cornflowerblue')
        if equal is not None:
            plt.scatter([equal.objs[0]], [equal.objs[1]], color='indianred')
        ax.view_init(10, -120)
        ax.set_xlabel(axis_titles[0], fontsize=12)
        ax.set_ylabel(axis_titles[1], fontsize=12)
        ax.set_zlabel(axis_titles[2], fontsize=12)
        ax.set_zlim3d([0, max(loss_dt[keys[2]])])
        plt.title(title, fontsize=14)

        plt.savefig(f"images/{title}.png", dpi=600, bbox_inches='tight')
        plt.show()

        return loss_dt

    def plot_three_objs_plotly(moo_objs, loss_dic=None, title=None, equal=None):

        if loss_dic is None:
            loss_dic = [{'key': f'obj{i}', 'title': f'Objective {i}'}\
                            for i in range(len(moo_objs[0]))]

        if title is None:
            title = 'Pareto Frontier'

        keys = [dic['key'] for dic in loss_dic]
        axis_titles = [dic['title'] for dic in loss_dic]
        
        loss = {keys[i]: [] for i in range(len(moo_objs[0]))}

        for solution in moo_objs:
            for i in range(len(moo_objs[0])):
                loss[keys[i]].append(solution[i])

        loss_dt = pd.DataFrame.from_dict(loss).sort_values(by=keys[0])
        x=loss_dt[keys[0]]
        y=loss_dt[keys[1]]
        z=loss_dt[keys[2]]

        # fig = plt.figure(figsize=(6,4))
        fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=5, color=z, colorscale='Viridis', opacity=0.8)
            )])
        
        fig.update_layout(
            title='Fronteira de Pareto (3D)',
            scene=dict(
                xaxis_title=axis_titles[0],
                yaxis_title=axis_titles[1],
                zaxis_title=axis_titles[2]
            ),
            font=dict(size=12)
        )
        fig.update_traces(marker=dict(line=dict(width=1, color='black'))) 
        fig.show()

        return loss_dt
    
    def plot_mu(importances):
        plt.plot(importances, color="purple", linewidth=2)
        plt.xlabel("Iterations")
        plt.ylabel("mu")
        plt.title(f"Evolution of margin along iteration")
        plt.legend()
        plt.grid()
        plt.show()