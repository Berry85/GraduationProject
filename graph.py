# -*- coding: utf-8 -*-

from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
import pandas as pd

FILE_PATH = "./Output/predicted.csv"

graph = Graph('http://localhost:7474', username='neo4j', password='sonic-senior-linda-sharp-vatican-9357')

file = pd.read_csv(FILE_PATH)
after_file = pd.read_csv("./Output/pre_after.csv")
concurrent_file = pd.read_csv("./Output/pre_concurrent.csv")


def split(x, label):
    # id, event_type, Age, Gender, Place, Object, Measurement_Result, Measurement_Prompt, procedure_occurrence_way, ConditionAge, DuringTime
    List = list(x)
    strings = str(List[0]).split(" ")
    if label == 'Event_Type':
        return strings[0]
    elif label == 'Age':
        return strings[1]
    elif label == 'Gender':
        return strings[2]
    elif label == 'Place':
        return strings[3]
    elif label == 'Object':
        return strings[4]
    elif label == 'Measurement_Result':
        return strings[5]
    elif label == 'Measurement_Prompt':
        return strings[6]
    elif label == 'procedure_occurrence_way':
        return strings[7]
    elif label == 'ConditionAge':
        return strings[8]
    elif label == 'DuringTime':
        return strings[9]
    elif label == 'After_Type':
        i = 10
        temp = ""
        if i <= len(List):
            temp = temp + strings[i]
            i = i + 1
        return temp


def read_data():
    file.rename(columns={'0': 'data'}, inplace=True)
    file['Event_Type'] = file.apply(lambda x: split(x, 'Event_Type'), axis=1)
    file['Age'] = file.apply(lambda x: split(x, 'Age'), axis=1)
    file['Gender'] = file.apply(lambda x: split(x, 'Gender'), axis=1)
    file['Place'] = file.apply(lambda x: split(x, 'Place'), axis=1)
    file['Object'] = file.apply(lambda x: split(x, 'Object'), axis=1)
    file['Measurement_Result'] = file.apply(lambda x: split(x, 'Measurement_Result'), axis=1)
    file['Measurement_Prompt'] = file.apply(lambda x: split(x, 'Measurement_Prompt'), axis=1)
    file['procedure_occurrence_way'] = file.apply(lambda x: split(x, 'procedure_occurrence_way'), axis=1)
    file['ConditionAge'] = file.apply(lambda x: split(x, 'ConditionAge'), axis=1)
    file['DuringTime'] = file.apply(lambda x: split(x, 'DuringTime'), axis=1)
    file.drop(columns=['data'], inplace=True)
    after_file.rename(columns={'0': 'After_Type'}, inplace=True)
    concurrent_file.rename(columns={'0': 'Concurrent_Type'}, inplace=True)


def MatchNode(m_graph, m_label, m_attrs):
    if m_label == "Place":
        m_n = "_.Place=" + "'" + m_attrs['Place'] + "'"
    elif m_label == "After_Event":
        m_n = "_.After_Type=" + "'" + m_attrs['After_Type'] + "'"
    elif m_label == "Concurrent_Event":
        m_n = "_.Concurrent_Type=" + "'" + m_attrs['Concurrent_Type'] + "'"
    else:
        m_n = "_.id=" + "'" + m_attrs['id'] + "'"
    matcher = NodeMatcher(m_graph)
    re_value = matcher.match(m_label).where(m_n).first()
    return re_value


def CreateRelationship(m_graph, m_label1, m_attrs1, m_label2, m_attrs2, m_r_name):
    reValue1 = MatchNode(m_graph, m_label1, m_attrs1)
    reValue2 = MatchNode(m_graph, m_label2, m_attrs2)
    if reValue1 is None or reValue2 is None:
        return False
    m_r = Relationship(reValue1, m_r_name, reValue2)
    n = graph.create(m_r)
    return n


def create_node(m_graph, m_label, m_attrs):
    if m_label == "Place":
        m_n = "_.Place=" + "\'" + m_attrs['Place'] + "\'"
    elif m_label == "After_Event":
        m_n = "_.After_Type=" + "\'" + m_attrs['After_Type'] + "\'"
    elif m_label == "Concurrent_Event":
        m_n = "_.Concurrent_Type=" + "'" + m_attrs['Concurrent_Type'] + "'"
    else:
        m_n = "_.id=" + "\'" + m_attrs['id'] + "\'"
    matcher = NodeMatcher(m_graph)
    re_value = matcher.match(m_label).where(m_n).first()
    m_node = Node(m_label, **m_attrs)
    if re_value is None:
        n = graph.create(m_node)
        return m_node
    return m_node


if __name__ == '__main__':
    read_data()
    CYPHER = "match (n) detach delete n "
    graph.run(CYPHER)
    CYPHER2 = "match (n) RETURN n"
    graph.run(CYPHER2)

    label1 = "Event"
    label2 = "Place"
    label3 = "Person"
    label4 = "After_Event"
    label5 = "Concurrent_Event"

    relation1 = "hasPlace"
    relation2 = "hasPerson"
    relation3 = "After_Type"
    relation4 = "Concurrent_Type"

    for i, j in file.iterrows():
        # Event
        attr1 = {"id": str(i + 1),
                 "Event_Type": j.Event_Type,
                 "Object": j.Object,
                 "Measurement_Result": j.Measurement_Result,
                 "Measurement_Prompt": j.Measurement_Prompt,
                 "procedure_occurrence_way": j.procedure_occurrence_way,

                 "DuringTime": j.DuringTime}
        event = create_node(graph, label1, attr1)

        # Place
        attr2 = {
            "Place": j.Place
        }
        place = create_node(graph, label2, attr2)

        # Person
        attr3 = {
            "id": str(i + 1),
            "Age": j.Age,
            "Gender": j.Gender,
            "ConditionAge": j.ConditionAge,
        }
        person = create_node(graph, label3, attr3)

        after_type = after_file.iloc[i]
        attr4 = {
            "After_Type": after_type.After_Type}
        after = create_node(graph, label4, attr4)

        concurrent_type = concurrent_file.iloc[i]
        attr5 = {
            "Concurrent_Type": concurrent_type.Concurrent_Type}
        concurrent = create_node(graph, label5, attr5)

        reValue1 = CreateRelationship(graph, label1, attr1, label2, attr2, relation1)
        reValue2 = CreateRelationship(graph, label1, attr1, label3, attr3, relation2)
        reValue3 = CreateRelationship(graph, label1, attr1, label4, attr4, relation3)
        reValue4 = CreateRelationship(graph, label1, attr1, label5, attr5, relation4)
        print(i)
