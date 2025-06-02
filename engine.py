import traci, random
from collections import defaultdict
from nn_architechure import *

sumo_binary = "sumo-gui"
sumo_config = "./cross_configuration/cross.sumocfg"

sumo_cmd = [sumo_binary, "-c", sumo_config, "--start", "--quit-on-end", "--seed", "42"]

# * Getter function
def get_state(edges: dict):
    # * Return [q_W, q_E, q_N, q_S]
    max_capacity = [816, 816, 408, 204]
    state = [] # W - E - N - S
    for _, lanes in edges.items():
        sum_halted = 0

        for lane in lanes:
            num_halted = traci.lane.getLastStepHaltingNumber(lane)
            sum_halted += num_halted

        state.append(sum_halted / len(lanes))

    for i in range(len(max_capacity)):
        state[i] /= max_capacity[i]

    return state

def get_reward(state):
    # * reward = - sum(state)
    return -sum(state)

def get_action(state: torch.Tensor, q_net, epsilon):
    '''
        epsilon greedy policy
    '''
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        q_net.eval()

        with torch.no_grad():
            q_values = q_net(state.unsqueeze(0))

        q_net.train()
        return q_values.argmax(dim=1).item()
        

def simulation_ep(num_simulates, q_net, buffer, decision_interval, yellow_interval, epsilon=0.1):
    traci.start(sumo_cmd)

    # * Traffic id and edge id
    tl = traci.trafficlight.getIDList()[0]
    lanes = set(traci.trafficlight.getControlledLanes(tl))
    edges = defaultdict(list)

    for lane in lanes:
        edge = lane.rsplit("_", 1)[0]
        edges[edge].append(lane)

    state = get_state(edges)
    current_phase = traci.trafficlight.getPhase(tl) # * traffic light at current time, 0 - NS green, 2 - EW green

    cumulative_reward = 0
    reward = 0
    action = 0

    sec = 0
    while sec < num_simulates:
        traci.simulationStep()
        sec += 1

        new_state = get_state(edges)
        # * Calculate cumulative reward
        cumulative_reward += get_reward(new_state)
        reward = reward + get_reward(new_state)
        # * Done flag
        done = (sec + 1) == num_simulates 

        if (sec + 1) % decision_interval == 0 or done:
            buffer.push(state, action, reward, new_state, done)
            # print(f"In time {sec + 1} : {state, action, reward:.2f, new_state, done}")    

            state = new_state
            reward = 0

            action = get_action(torch.tensor(state, dtype=torch.float32), q_net, epsilon=epsilon)

            # * Apply action
            if action:
                # * Transition time
                yellow_phase = current_phase + 1
                traci.trafficlight.setPhase(tl, yellow_phase)

                for _ in range(yellow_interval):
                    sec += 1
                    traci.simulationStep()

                current_phase = 2 if current_phase == 0 else 0  # * Toggle phase
                traci.trafficlight.setPhase(tl, current_phase)

    traci.close()
    
    return cumulative_reward

def train_step(buffer: ReplayBuffer, q_net: torch.nn.Module, target_net: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, batch_size: int, gamma: int):
    
    if len(buffer) < batch_size:
        print("Error, dont have enough samples")
        return
    
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    states      = states.to(device)
    actions     = actions.to(device)
    rewards     = rewards.to(device)
    next_states = next_states.to(device)
    dones       = dones.to(device)

    q_values = q_net(states)
    q_values = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1)[0]
        target_q = rewards + gamma * next_q_values * (1 - dones)

    loss = F.mse_loss(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def update_target(q_net: torch.nn.Module, target_net: torch.nn.Module):
    target_net.load_state_dict(q_net.state_dict())

def build_state_str(lanes):
    we_green, sn_green, we_yellow, sn_yellow = "", "", "", ""

    for lane in lanes:
        if lane.startswith("3T2_W2C") or lane.startswith("3T2_E2C"):
            we_green  += "g"
            we_yellow += "y"
            sn_green  += "r"
            sn_yellow += "r"

        else:
            we_green  += "r"
            we_yellow += "r"
            sn_green  += "g"
            sn_yellow += "y"

    return {
        "WE_green":  we_green,
        "SN_green":  sn_green,
        "WE_yellow": we_yellow,
        "SN_yellow": sn_yellow
    }


def static_traffic(num_simulates, t_we_green=40, t_sn_green=20, t_yellow=3, collect=False):
    '''
        40s green for WE (3T2 street), 20s green for NS (CT street), 3s yellow
    '''
    traci.start(sumo_cmd)

    tl = traci.trafficlight.getIDList()[0]
    lanes = list(traci.trafficlight.getControlledLanes(tl))
    traffic_states = build_state_str(lanes)

    unique_lanes = set(traci.trafficlight.getControlledLanes(tl))
    edges = defaultdict(list)

    for lane in unique_lanes:
        edge = lane.rsplit("_", 1)[0]
        edges[edge].append(lane)

    phases = [
        ("WE_green" , traffic_states["WE_green"] , t_we_green),
        ("WE_yellow", traffic_states["WE_yellow"], t_yellow),
        ("SN_green" , traffic_states["SN_green"] , t_sn_green),
        ("SN_yellow", traffic_states["SN_yellow"], t_yellow)
    ]

    t = 0
    cumulative_reward = 0

    while t < num_simulates:
        for phase_name, state_str, duration in phases:
            if t >= num_simulates:
                break
        
            traci.trafficlight.setRedYellowGreenState(tl, state_str)

            for _ in range(duration):
                if t >= num_simulates:
                    break

                traci.simulationStep()
                t += 1

                state = get_state(edges)
                current_reward = get_reward(state)
                cumulative_reward += current_reward

    traci.close()

    print(f"Number simulation: {num_simulates}s")
    print(f"Cumulative reward through episode: {cumulative_reward:.2f}")

if __name__ == "__main__":
    # * Simulation config
    num_simulates = 3600 
    decision_interval = 15 
    yellow_interval = 3 

    # * Neuron network configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 4
    hidden_dim = [256, 256, 256]
    action_dim = 2

    # * Init network
    Q_network = FCNetwork(state_dim, hidden_dim, action_dim, device).to(device)
    # target_network = FCNetwork(state_dim, hidden_dim, action_dim, device).to(device)
    # target_network.load_state_dict(Q_network.state_dict())
    # target_network.eval()

    # * Init replay buffer
    buffer = ReplayBuffer(100000)

    optimizer = torch.optim.Adam(Q_network.parameters(), lr=1e-3)
    gamma = 0.95

    epsilon_start = 0.1
    epsilon_end   = 0.01    
    decay_rate    = 0.995

    epsilon = epsilon_start

    num_epochs = 5
    num_episodes = 500

    log_path = "training_log.txt"

    # print("Train - section:")
    # with open(log_path, "w") as log_file:
    #     for ep in range(num_episodes):
    #         buffer.clear()
    #         epsilon = max(epsilon_end, epsilon * decay_rate)
    #         r = simulation_ep(num_simulates=num_simulates, q_net=Q_network, buffer=buffer, 
    #                     decision_interval=decision_interval, yellow_interval=yellow_interval, epsilon=epsilon)
    #         print(f"Episode {ep + 1}: reward = {r:.2f}")
    #         log_file.write(f"Episode {ep + 1}: reward = {r:.2f}\n")

    #         for epoch in range(num_epochs):
    #             loss = train_step(buffer=buffer, q_net=Q_network, target_net=target_network, optimizer=optimizer, gamma=gamma, batch_size=len(buffer))
    #             log_file.write(f"    Epoch {epoch + 1}: loss = {loss:.2f}\n")
    #             print(f"    Epoch {epoch + 1}: loss = {loss:.2f}")

    #         update_target(q_net=Q_network, target_net=target_network)

    #         if (ep + 1) % 10 == 0:
    #             torch.save(Q_network.state_dict(), f"q_network_ep{ep+1}.pt")
    
    checkpoint = torch.load("q_network_2.pt", map_location=device)
    Q_network.load_state_dict(checkpoint)
    Q_network.eval()

    print("Test - section:")
    r = simulation_ep(num_simulates=num_simulates, q_net=Q_network, buffer=buffer, 
            decision_interval=decision_interval, yellow_interval=yellow_interval, epsilon=0)
    print(f"Reward = {r:.2f}")

    print("Static - section:")
    static_traffic(num_simulates)
