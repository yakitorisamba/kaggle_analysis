import xml.etree.ElementTree as ET

from typing import Dict, List, Set

import pandas as pd

from dataclasses import dataclass

from collections import defaultdict


@dataclass

class Node:

    id: str

    type: str

    next_nodes: List[str]

    properties: Dict

    

class TableauFlowParser:

    def __init__(self, xml_path: str):

        self.tree = ET.parse(xml_path)

        self.root = self.tree.getroot()

        self.nodes: Dict[str, Node] = {}

        self.initial_nodes: List[str] = []

        self.generated_code: List[str] = []

        

    def parse_nodes(self):

        """XMLからノード情報を解析"""

        for node in self.root.findall(".//node"):

            node_id = node.get('id')

            node_type = node.get('type')

            

            # 次のノードを取得

            next_nodes = []

            for connection in self.root.findall(f".//connection[@source='{node_id}']"):

                next_nodes.append(connection.get('target'))

            

            # ノードのプロパティを取得

            properties = {}

            for prop in node.findall('./properties/*'):

                properties[prop.tag] = prop.text

                

            self.nodes[node_id] = Node(node_id, node_type, next_nodes, properties)

            

            # 初期ノードを特定

            if node_type == 'initial':

                self.initial_nodes.append(node_id)

    

    def generate_node_code(self, node: Node) -> str:

        """各ノードタイプに応じたPythonコードを生成"""

        if node.type == 'rename':

            columns = eval(node.properties.get('columns', '{}'))

            return f"df{node.id} = df.rename(columns={columns})"

            

        elif node.type == 'filter':

            condition = node.properties.get('condition')

            return f"df{node.id} = df[{condition}]"

            

        elif node.type == 'aggregate':

            group_by = eval(node.properties.get('group_by', '[]'))

            aggs = eval(node.properties.get('aggregations', '{}'))

            return f"df{node.id} = df.groupby({group_by}).agg({aggs})"

            

        elif node.type == 'join':

            join_type = node.properties.get('join_type', 'inner')

            on = eval(node.properties.get('on', '[]'))

            return f"df{node.id} = pd.merge(left_df, right_df, how='{join_type}', on={on})"

            

        # 他のノードタイプに応じて追加

        return f"# Unhandled node type: {node.type}"

    

    def generate_code(self) -> str:

        """フロー全体のPythonコードを生成"""

        visited = set()

        

        def traverse_nodes(node_id: str):

            if node_id in visited:

                return

            

            visited.add(node_id)

            node = self.nodes[node_id]

            

            # コード生成

            code = self.generate_node_code(node)

            self.generated_code.append(code)

            

            # 次のノードを処理

            for next_node in node.next_nodes:

                traverse_nodes(next_node)

        

        # 初期ノードから処理開始

        for initial_node in self.initial_nodes:

            traverse_nodes(initial_node)

            

        return "\n".join([

            "import pandas as pd",

            "import numpy as np",

            "",

            "# Generated code from Tableau Prep Flow",

            *self.generated_code

        ])

    

    def parse_and_generate(self) -> str:

        """パース処理とコード生成を実行"""

        self.parse_nodes()

        return self.generate_code()


# 使用例

def convert_tfl_to_python(xml_path: str) -> str:

    parser = TableauFlowParser(xml_path)

    return parser.parse_and_generate()

