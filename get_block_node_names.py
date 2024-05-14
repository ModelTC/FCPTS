import torch
from torchvision.models.feature_extraction import get_graph_node_names
from torch.fx import symbolic_trace
from torch.fx import GraphModule, Node

def find_num_nodes(nodes):
    num = 0
    for node in nodes:
        if isinstance(node, Node):
            num += 1
    return num


def extract_layer(node, fp32_modules):
    _SUPPORT_MODULE_TYPES = (torch.nn.Conv2d, torch.nn.Linear)
    layer_node_list = []
    cur_node = node
    is_next_block = False  # check whether stoped by a block
    while True:
        print('cur_node in layer is {}'.format(cur_node))
        layer_node_list.append(cur_node)  # valid node here
        stop = (len(cur_node.users) == 0)
        for user in cur_node.users:
            if user.op == 'call_module' and isinstance(fp32_modules[user.target], _SUPPORT_MODULE_TYPES):
                stop = True
            # TODO: only short-cut here, consider more here
            # TODO: can also use un/completed to check here.
            if ('add' in user.name and user.op in ['call_function', 'call_method']):
                stop = True
            if ('cat' in user.name and user.op in ['call_function', 'call_method']):
                stop = True
            if user.op == 'output':
                is_next_block, stop = True, True
        if stop:
            break
        cur_node = list(cur_node.users.keys())[0]
    if find_num_nodes(cur_node.users) > 1:
        is_next_block = True
    return layer_node_list, is_next_block



# Recommend: log this to check if the block is right. You can define your own block manually or automatically like this
# extract the block one such as short-cut
def extract_block(input_nodes, fp32_modules, depth=0):
    if depth > 2:
        # stack 2 or 3 layers for no short-cut structure
        return []
    layer_node_list = []
    is_block = False
    cnt = dict()
    q, p = [], []  # q records the completed node, p records the uncompleted nodes
    for input in input_nodes:
        for user in input.users:
            if user not in cnt:
                cnt[user] = find_num_nodes(user.args)
                if cnt[user] > 1:
                    is_block = True
                p.append(user)
            cnt[user] -= 1
            if cnt[user] == 0:
                q.append(user)
                p.remove(user)
    
    while len(q) != 0:
        cur_node = q.pop(0)  # valid node here
        print('cur node is {}'.format(cur_node))
        if len(p) == 0 and len(q) == 0:
            break
        layer_node_list.append(cur_node)
        for user in cur_node.users:
            if user not in cnt:
                cnt[user] = find_num_nodes(user.args)
                if cnt[user] > 1:
                    is_block = True
                p.append(user)
            cnt[user] -= 1
            if cnt[user] == 0:
                q.append(user)
                p.remove(user)
        print('uncompleted nodes are {}'.format(p))
    exp_nodes, is_next_block = extract_layer(cur_node, fp32_modules)
    if is_block or is_next_block:
        return layer_node_list + exp_nodes
    else:
        return layer_node_list + exp_nodes + extract_block([exp_nodes[-1]], fp32_modules, depth + 1)

def find_cur_node(layer_node_list):
    print("layer_node_list is ", layer_node_list)
    for node in reversed(layer_node_list):
        if node.target == 'update':
            continue
        if isinstance(node.target, str) and 'const' in node.target:
            continue
        if node.op == 'call_method' or node.op == 'call_function':
            continue
        return node
    raise ValueError('Bad layer node list provided.')



def get_block_node_names(model_traced, fp32_modules):
    _SUPPORT_MODULE_TYPES = (torch.nn.Conv2d, torch.nn.Linear)

    nodes = list(model_traced.graph.nodes)


    block_nodes = []

    checked_nodes = dict()
    for node in nodes:
        if node in checked_nodes:
            continue
        if node.op == "call_module" and isinstance(fp32_modules[node.target], _SUPPORT_MODULE_TYPES):
            # layer_node_list, is_next_block = extract_layer(node, fp32_modules)
            layer_node_list = extract_block(node.all_input_nodes, fp32_modules)
            # print("------------------- ", layer_node_list)

            cur_node = find_cur_node(layer_node_list)
            # print("cur_node : ", cur_node)

            block_nodes.append(cur_node)

            for x in layer_node_list:
                checked_nodes[x] = True

    # print("block_nodes : ", block_nodes)

    block_node_names = [node.target for node in block_nodes]
    # print("block_node_names : ", block_node_names)

    return block_node_names