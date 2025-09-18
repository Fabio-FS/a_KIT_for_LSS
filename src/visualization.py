# visualization_fixed.py
import plotly.graph_objects as go
import json
import os

def create_interactive_conversation_tree_with_agent_selection(
    posts_data,
    graph_data,
    individual_likes_data,
    max_timesteps=100,
    spine_x=0,
    time_spacing=2,
    read_spacing=1.5,
    read_offset=1,
    window_size=20
):
    """
    Interactive conversation tree with TWO independent sliders:
      • Agent slider toggles which agent's traces are visible.
      • Time slider only changes the y-axis range/ticks (window), preserving agent selection.

    Visual tweaks:
      • Edges are always drawn UNDER nodes.
      • Read nodes sit slightly BELOW the spine of the same timestep to suggest direction,
        but the window boundaries are chosen so that future/past reads never leak in.
    """

    # ----------------------------
    # Tunable offsets
    # ----------------------------
    READ_VOFFSET_FRACTION = 0.50  # how far below the spine to place read nodes (0..1)
    EPS_FRACTION = 0.15           # small padding to avoid clipping markers

    # ----------------------------
    # Helpers
    # ----------------------------
    # Build posts lookup
    posts_lookup = {(agent_id, timestep): text for agent_id, timestep, text in posts_data}

    num_agents = len(graph_data["agents"])
    fig = go.Figure()

    def format_persona_info(agent_data):
        lines = [
            f"<b>{agent_data['name']}</b>",
            f"<b>Big 5 Traits:</b>",
            f"• Openness: {agent_data.get('openness', 'N/A')}/100",
            f"• Conscientiousness: {agent_data.get('conscientiousness', 'N/A')}/100",
            f"• Extraversion: {agent_data.get('extraversion', 'N/A')}/100",
            f"• Agreeableness: {agent_data.get('agreeableness', 'N/A')}/100",
            f"• Neuroticism: {agent_data.get('neuroticism', 'N/A')}/100",
            f"<b>Political Views:</b>",
            f"• Economic L/R: {agent_data.get('economic_left_right', 'N/A')}/100",
            f"• Social C/L: {agent_data.get('social_conservative_liberal', 'N/A')}/100",
        ]
        return "<br>".join(lines)

    def wrap_text(text, max_length=50):
        if len(text) <= max_length:
            return text
        words = text.split()
        lines, current, length = [], [], 0
        for w in words:
            if current and length + len(w) + 1 > max_length:
                lines.append(" ".join(current))
                current, length = [w], len(w)
            else:
                current.append(w)
                length += len(w) + 1
        if current:
            lines.append(" ".join(current))
        return "<br>".join(lines)

    def get_agent_color(agent_data):
        econ = agent_data.get("economic_left_right", 50)
        if econ < 33:
            return "blue"
        elif econ > 67:
            return "red"
        return "purple"

    def agent_liked_post(timeline_agent_id, author_id, post_timestep, current_timestep, individual_likes_data):
        if post_timestep >= individual_likes_data.shape[2]:
            return False
        
        return individual_likes_data[timeline_agent_id][author_id][post_timestep]


    def get_like_color(timeline_agent_id, author_id, post_timestep, current_timestep):
        if agent_liked_post(timeline_agent_id, author_id, post_timestep, current_timestep, individual_likes_data):
            return "green"
        else:
            return "lightgray"
    # ----------------------------
    # Build traces PER AGENT over the full time axis (absolute y = t * time_spacing)
    # ----------------------------
    agent_traces_indices = {aid: [] for aid in range(num_agents)}
    max_total_timesteps = 0
    read_voffset = READ_VOFFSET_FRACTION * time_spacing

    for selected_agent_id in range(num_agents):
        read_history = graph_data["read_history"][selected_agent_id]
        agent_data = graph_data["agents"][selected_agent_id]

        total_timesteps = min(max_timesteps, len(read_history))
        max_total_timesteps = max(max_total_timesteps, total_timesteps)

        # -------- collect geometry --------
        # spine nodes (one per timestep)
        spine_nodes = []
        for t in range(total_timesteps):
            spine_nodes.append({
                "x": spine_x,
                "y": t * time_spacing,  # align exactly to the timestep line
                "agent_id": selected_agent_id,
                "agent_data": agent_data,
                "timestep": t,
                "text": posts_lookup.get((selected_agent_id, t), "[No post]"),
            })

        # spine edges (multi-segment)
        spine_edges_x, spine_edges_y = [], []
        for t in range(total_timesteps - 1):
            spine_edges_x.extend([spine_x, spine_x, None])
            spine_edges_y.extend([t * time_spacing, (t + 1) * time_spacing, None])

        # read nodes + edges (slightly below the same timestep line)
        read_nodes_x, read_nodes_y, read_nodes_text, read_nodes_custom, like_colors = [], [], [], [], []
        read_edges_x, read_edges_y = [], []

        for t in range(total_timesteps):
            if t >= len(read_history):
                continue
            reads = read_history[t]
            for i, (author_id, post_t) in enumerate(reads):
                if author_id == -1 or post_t == -1:
                    continue
                author_data = graph_data["agents"][author_id]
                post_text = posts_lookup.get((author_id, post_t), "[Post not found]")

                x_pos = spine_x + read_offset + (i * read_spacing)
                y_pos = t * time_spacing - read_voffset  # below the spine but same timestep

                read_nodes_x.append(x_pos)
                read_nodes_y.append(y_pos)
                read_nodes_text.append(str(author_id))
                read_nodes_custom.append([format_persona_info(author_data), post_t, wrap_text(post_text)])
                like_colors.append(get_like_color(selected_agent_id, author_id, post_t, t))

                # edge from read node to spine node (same timestep line)
                read_edges_x.extend([x_pos, spine_x, None])
                read_edges_y.extend([y_pos, t * time_spacing, None])

        # -------- add traces in this order: edges first (under), nodes last (over) --------
        # 1) spine edges
        if spine_edges_x:
            fig.add_trace(go.Scatter(
                x=spine_edges_x, y=spine_edges_y, mode="lines",
                line=dict(color="gray", width=2),
                hoverinfo="skip", showlegend=False,
                visible=(selected_agent_id == 0),
                name=f"Agent {selected_agent_id} spine edges",
            ))
            agent_traces_indices[selected_agent_id].append(len(fig.data) - 1)

        # 2) read edges
        if read_edges_x:
            fig.add_trace(go.Scatter(
                x=read_edges_x, y=read_edges_y, mode="lines",
                line=dict(color="gray", width=2),
                hoverinfo="skip", showlegend=False,
                visible=(selected_agent_id == 0),
                name=f"Agent {selected_agent_id} read edges",
            ))
            agent_traces_indices[selected_agent_id].append(len(fig.data) - 1)

        # 3) spine nodes (on top of edges)
        if spine_nodes:
            fig.add_trace(go.Scatter(
                x=[n["x"] for n in spine_nodes],
                y=[n["y"] for n in spine_nodes],
                mode="markers+text",
                marker=dict(symbol="square", size=25,
                            color=get_agent_color(agent_data),
                            line=dict(color="darkgreen", width=2)),
                text=[str(n["agent_id"]) for n in spine_nodes],
                textposition="middle center",
                textfont=dict(color="white", size=10),
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "<b>Post (t=%{customdata[1]}):</b> %{customdata[2]}<br>"
                    "<extra></extra>"
                ),
                customdata=[[format_persona_info(n["agent_data"]), n["timestep"], wrap_text(n["text"])]
                            for n in spine_nodes],
                showlegend=False, visible=(selected_agent_id == 0),
                name=f"Agent {selected_agent_id} spine",
            ))
            agent_traces_indices[selected_agent_id].append(len(fig.data) - 1)

        # 4) read nodes (on very top)
        if read_nodes_x:
            fig.add_trace(go.Scatter(
                x=read_nodes_x, y=read_nodes_y, mode="markers+text",
                marker=dict(symbol="circle", size=20, color=like_colors,
                            line=dict(color="blue", width=2), opacity=0.85),
                text=read_nodes_text,
                textposition="middle center",
                textfont=dict(color="black", size=8),
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "<b>Post (t=%{customdata[1]}):</b> %{customdata[2]}<br>"
                    "<extra></extra>"
                ),
                customdata=read_nodes_custom,
                showlegend=False, visible=(selected_agent_id == 0),
                name=f"Agent {selected_agent_id} reads",
            ))
            agent_traces_indices[selected_agent_id].append(len(fig.data) - 1)

    # ----------------------------
    # Sliders
    # ----------------------------
    # Agent slider – toggles visibility among agent trace groups
    agent_steps = []
    for agent_id in range(num_agents):
        visibility = [False] * len(fig.data)
        for idx in agent_traces_indices[agent_id]:
            visibility[idx] = True

        agent_data = graph_data["agents"][agent_id]
        agent_steps.append(dict(
            method="update",
            args=[
                {"visible": visibility},  # only change visibility
                {"title": f"Agent {agent_id} - {agent_data['name']}"},
            ],
            label=f"{agent_id}: {agent_data['name']}",
        ))

    # Time slider – ONLY changes the y-axis range and tick labels; never touches visibility
    max_total_timesteps = max(1, max_total_timesteps)
    max_window_starts = max(1, max_total_timesteps - window_size + 1)
    eps = EPS_FRACTION * time_spacing

    time_steps = []
    for window_start in range(max_window_starts):
        y0 = window_start * time_spacing - read_voffset - eps
        y1 = (window_start + window_size - 1) * time_spacing + eps
        time_steps.append(dict(
            method="relayout",
            args=[{
                "yaxis.range": [y0, y1],
                "yaxis.tickvals": [((window_start + k) * time_spacing) for k in range(window_size)],
                "yaxis.ticktext": [f"t={window_start + k}" for k in range(window_size)],
            }],
            label=f"t={window_start}-{window_start + window_size - 1}",
        ))

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Agent: "},
            pad={"t": 100},
            steps=agent_steps,
            x=0.1, len=0.35, y=-0.1,
        ),
        dict(
            active=0,
            currentvalue={"prefix": "Time: "},
            pad={"t": 100},
            steps=time_steps,
            x=0.55, len=0.35, y=-0.1,
        ),
    ]

    # ----------------------------
    # Layout
    # ----------------------------
    # Initial y-range corresponds to window starting at 0 and includes the lowered reads
    eps = EPS_FRACTION * time_spacing
    init_y0 = 0 * time_spacing - read_voffset - eps
    init_y1 = (window_size - 1) * time_spacing + eps

    fig.update_layout(
        title=f"Agent 0 - {graph_data['agents'][0]['name']}",
        xaxis_title="",
        yaxis_title="Time →",
        showlegend=False,
        hovermode="closest",
        width=800,
        height=900,
        margin=dict(b=150),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            range=[init_y0, init_y1],
            tickvals=[k * time_spacing for k in range(window_size)],
            ticktext=[f"t={k}" for k in range(window_size)],
            tickfont=dict(size=10),
        ),
        plot_bgcolor="white",
        sliders=sliders,
    )

    return fig


def show_interactive_conversation_tree_with_agent_selection(replica_dir, max_timesteps=100, window_size=5):
    from .save import load_replica

    replica_data = load_replica(replica_dir)

    graph_path = os.path.join(replica_dir, "graph_data.json")
    with open(graph_path, "r") as f:
        graph_data = json.load(f)

    fig = create_interactive_conversation_tree_with_agent_selection(
        replica_data["POSTS_long"],
        graph_data,
        replica_data["INDIVIDUAL_LIKES"],  # Use new individual likes data
        max_timesteps=max_timesteps,
        window_size=window_size,
    )

    fig.show()
    return fig
