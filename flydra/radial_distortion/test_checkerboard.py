import numpy as np
from simplenx import Graph
import checkerboard

class TestCheckerboard:
    def setUp(self):
        self.x = np.array([
        34.5,   51.5,   71. ,   45. ,   99. ,   92.5,   64. ,   38. ,
         57. ,   31. ,   85.5,   78.5,   51.5,   26.5,   46.5,   22.5,
         75. ,   72.5,   45.5,   20.5,   43.5,   18. ,   71. ,   69.5,
         42.5,   17. ,  102.5,  100.5,   99.5,  133. ,  131.5,  135. ,
        171. ,  169.5,  167. ,  208. ,  205.5,  210. ,  251.5,  249.5,
        247.5,  294. ,  292.5,  296.5,  343.5,  340.5,  339.5,  389. ,
        387. ,  391.5,  440.5,  438.5,  436. ,  487.5,  485. ,  489.5,
        537.5,  535.5,  532.5,  581.5,  579. ,  583.5,  627.5,  626. ,
        622.5,  666.5,  663.5,  669. ,  707. ,  627.5,  668.5,  707. ,
        741.5,  743. ,  665.5,  703.5,  739.5,  700. ,  734. ,  659. ,
        662.5,  616.5,  655. ,  620. ,  622.5,  625.5,  578. ,  580.5,
        582. ,  583.5,  536. ,  537. ,  538. ,  490.5,  490. ,  444. ,
        490. ,  443.5,  442. ,  396. ,  394. ,  351.5,  398. ,  348.5,
        346. ,  304. ,  300. ,  264.5,  307. ,  260.5,  255. ,  213.5,
        219. ,  174. ,  224.5,  180. ,  185.5,  228. ,  268.5,  138. ,
        143.5,  150. ,  155.5,  190.5,  109.5,  116.5,  161.5,  122.5,
        128.5,  233. ,  195.5,  272.5,  310.5,  355. ,  400.5,  399. ,
        445. ,  444.5,  489. ,  490. ,  533. ,  534.5,  575.5,  106. ,
        104.5])
        self.y = np.array([
        5.5,   10.5,   24. ,   26.5,   21.5,   38.5,   41. ,   42.5,
         58.5,   59.5,   57. ,   76. ,   76.5,   77. ,   96. ,   95. ,
         95. ,  115.5,  114.5,  114.5,  134.5,  133.5,  135. ,  155. ,
        153.5,  152. ,  115.5,  136.5,  156.5,  137.5,  158.5,  115.5,
        116.5,  138.5,  160.5,  139.5,  163. ,  116.5,  117.5,  141. ,
        164. ,  142. ,  165.5,  118. ,  119. ,  144. ,  167.5,  144.5,
        168.5,  119. ,  120.5,  145. ,  169.5,  146.5,  170.5,  120. ,
        122.5,  146.5,  171.5,  147.5,  171.5,  123.5,  124. ,  148. ,
        171.5,  148.5,  171.5,  125. ,  126. ,  100.5,  101.5,  103. ,
        104. ,  126.5,   79.5,   81.5,   82.5,   60. ,   62.5,   37. ,
         58. ,   15. ,   17.5,   35. ,   56. ,   78. ,   33. ,   54.5,
         76.5,   99.5,   53. ,   75.5,   97.5,   74. ,   97. ,   51.5,
         52. ,   73. ,   96.5,   72.5,   96. ,   51. ,   50.5,   72. ,
         95.5,   72. ,   95. ,   51.5,   50. ,   73. ,   95.5,   93.5,
         73. ,   95. ,   52.5,   73. ,   53. ,   32.5,   30.5,   94. ,
         74. ,   54. ,   35.5,   33.5,   74.5,   55.5,   17. ,   37. ,
         19.5,   12.5,   14.5,   11. ,   30. ,   30. ,    9. ,   29.5,
          9. ,   29.5,   10. ,   30.5,   11. ,   31.5,   12.5,    3.5,
         94.5])
        self.aspect_ratio = 0.5

        self.expected_graph = Graph({47: {45: None, 51: None, 49: None, 48: None},
    23: {24: None, 28: None, 22: None}, 122: {123: None, 127: None,
    126: None, 121: None}, 34: {30: None, 36: None, 33: None}, 27:
    {22: None, 29: None, 26: None, 28: None}, 85: {74: None, 88: None,
    84: None, 69: None}, 46: {42: None, 48: None, 45: None}, 132:
    {118: None, 133: None, 108: None}, 64: {60: None, 66: None, 63:
    None}, 69: {70: None, 89: None, 85: None, 62: None}, 128: {4:
    None, 126: None, 127: None}, 82: {81: None, 79: None}, 6: {5:
    None, 7: None, 8: None, 2: None}, 55: {50: None, 56: None, 53:
    None, 94: None}, 35: {33: None, 39: None, 36: None, 37: None},
    104: {100: None, 106: None, 103: None, 44: None}, 41: {39: None,
    45: None, 42: None, 43: None}, 22: {20: None, 27: None, 23: None,
    17: None}, 66: {64: None, 65: None}, 51: {47: None, 53: None, 52:
    None, 50: None}, 91: {88: None, 93: None, 92: None, 90: None}, 9:
    {8: None, 7: None, 13: None}, 60: {58: None, 64: None, 59: None},
    28: {23: None, 30: None, 27: None}, 77: {80: None, 78: None, 75:
    None}, 84: {80: None, 87: None, 85: None, 83: None}, 70: {69:
    None, 71: None, 74: None, 67: None}, 119: {113: None, 144: None,
    120: None, 31: None}, 127: {5: None, 122: None, 128: None, 125:
    None}, 11: {12: None, 124: None, 16: None, 10: None}, 0: {1: None,
    3: None}, 29: {27: None, 33: None, 31: None, 30: None}, 24: {23:
    None, 25: None, 20: None}, 49: {44: None, 50: None, 47: None, 100:
    None}, 2: {4: None, 3: None, 6: None}, 42: {40: None, 46: None,
    41: None}, 56: {55: None, 61: None, 92: None, 57: None}, 133:
    {132: None, 135: None, 101: None}, 74: {75: None, 85: None, 70:
    None, 80: None}, 36: {34: None, 40: None, 35: None}, 141: {86:
    None, 139: None, 140: None, 90: None}, 8: {9: None, 10: None, 12:
    None, 6: None}, 78: {77: None, 76: None}, 59: {57: None, 63: None,
    60: None, 61: None}, 16: {14: None, 144: None, 11: None, 17:
    None}, 15: {14: None, 13: None, 19: None}, 83: {79: None, 86:
    None, 81: None, 84: None}, 25: {24: None, 21: None}, 120: {115:
    None, 124: None, 119: None, 121: None}, 57: {53: None, 59: None,
    56: None, 58: None}, 14: {15: None, 16: None, 18: None, 12: None},
    61: {56: None, 62: None, 59: None, 89: None}, 106: {104: None,
    110: None, 105: None, 43: None}, 43: {38: None, 44: None, 106:
    None, 41: None}, 1: {0: None, 143: None, 3: None}, 134: {136:
    None, 135: None}, 37: {32: None, 38: None, 111: None, 35: None},
    20: {21: None, 22: None, 18: None, 24: None}, 30: {28: None, 34:
    None, 29: None}, 21: {20: None, 19: None, 25: None}, 79: {83:
    None, 82: None, 80: None}, 7: {6: None, 3: None, 9: None}, 39:
    {35: None, 41: None, 40: None, 38: None}, 72: {71: None, 76: None,
    73: None}, 101: {108: None, 102: None, 133: None, 103: None}, 65:
    {63: None, 67: None, 66: None}, 50: {49: None, 55: None, 98: None,
    51: None}, 75: {74: None, 76: None, 71: None, 77: None}, 103: {99:
    None, 105: None, 104: None, 101: None}, 58: {54: None, 60: None,
    57: None}, 121: {116: None, 125: None, 122: None, 120: None}, 107:
    {108: None, 114: None, 109: None, 118: None}, 125: {10: None, 121:
    None, 127: None, 124: None}, 139: {137: None, 141: None, 96: None,
    138: None}, 44: {43: None, 49: None, 45: None, 104: None}, 112:
    {109: None, 115: None, 111: None, 114: None}, 26: {17: None, 31:
    None, 144: None, 27: None}, 38: {37: None, 43: None, 39: None,
    110: None}, 115: {112: None, 120: None, 116: None, 113: None},
    143: {1: None, 126: None, 4: None}, 99: {97: None, 103: None, 102:
    None, 100: None}, 136: {134: None, 138: None, 137: None}, 71: {70:
    None, 72: None, 75: None, 68: None}, 129: {131: None, 130: None,
    117: None}, 62: {61: None, 67: None, 63: None, 69: None}, 89: {69:
    None, 92: None, 88: None, 61: None}, 10: {8: None, 125: None, 11:
    None, 5: None}, 126: {130: None, 143: None, 122: None, 128: None},
    135: {133: None, 137: None, 134: None, 102: None}, 3: {2: None, 0:
    None, 7: None, 1: None}, 81: {142: None, 82: None, 83: None}, 144:
    {16: None, 119: None, 26: None, 124: None}, 4: {2: None, 128:
    None, 143: None, 5: None}, 13: {12: None, 9: None, 15: None}, 52:
    {48: None, 54: None, 51: None}, 45: {41: None, 47: None, 46: None,
    44: None}, 124: {11: None, 120: None, 144: None, 125: None}, 40:
    {36: None, 42: None, 39: None}, 95: {96: None, 102: None, 137:
    None, 97: None}, 111: {110: None, 113: None, 112: None, 37: None},
    76: {75: None, 72: None, 78: None}, 88: {85: None, 91: None, 87:
    None, 89: None}, 113: {111: None, 119: None, 32: None, 115: None},
    94: {92: None, 98: None, 93: None, 55: None}, 137: {135: None,
    139: None, 95: None, 136: None}, 98: {94: None, 100: None, 50:
    None, 97: None}, 131: {129: None, 118: None}, 80: {77: None, 84:
    None, 74: None, 79: None}, 130: {126: None, 129: None, 123: None},
    140: {138: None, 142: None, 141: None}, 108: {101: None, 107:
    None, 105: None, 132: None}, 123: {117: None, 122: None, 130:
    None, 116: None}, 93: {91: None, 97: None, 96: None, 94: None},
    18: {17: None, 19: None, 14: None, 20: None}, 116: {114: None,
    121: None, 123: None, 115: None}, 87: {84: None, 90: None, 88:
    None, 86: None}, 53: {51: None, 57: None, 54: None, 55: None}, 48:
    {46: None, 52: None, 47: None}, 54: {52: None, 58: None, 53:
    None}, 102: {95: None, 101: None, 99: None, 135: None}, 63: {59:
    None, 65: None, 64: None, 62: None}, 100: {98: None, 104: None,
    49: None, 99: None}, 67: {62: None, 68: None, 70: None, 65: None},
    109: {105: None, 112: None, 110: None, 107: None}, 90: {87: None,
    96: None, 141: None, 91: None}, 118: {117: None, 132: None, 131:
    None, 107: None}, 96: {90: None, 95: None, 93: None, 139: None},
    33: {29: None, 35: None, 32: None, 34: None}, 142: {81: None, 140:
    None, 86: None}, 5: {127: None, 6: None, 4: None, 10: None}, 138:
    {136: None, 140: None, 139: None}, 31: {26: None, 32: None, 119:
    None, 29: None}, 19: {18: None, 15: None, 21: None}, 17: {18:
    None, 26: None, 22: None, 16: None}, 32: {31: None, 37: None, 113:
    None, 33: None}, 12: {11: None, 13: None, 8: None, 14: None}, 97:
    {93: None, 99: None, 95: None, 98: None}, 117: {123: None, 118:
    None, 129: None, 114: None}, 92: {89: None, 94: None, 56: None,
    91: None}, 68: {67: None, 73: None, 71: None}, 73: {68: None, 72:
    None}, 110: {106: None, 111: None, 109: None, 38: None}, 105:
    {103: None, 109: None, 106: None, 108: None}, 86: {83: None, 141:
    None, 87: None, 142: None}, 114: {107: None, 116: None, 112: None,
    117: None}})
        self.expected_subgraphs=[
            Graph({0: {1: None}, 1: {0: None}}),

            Graph({1: {3: None}, 9: {13: None, 7: None}, 25: {21:
            None}, 15: {19: None, 13: None}, 7: {9: None, 3: None},
            21: {19: None, 25: None}, 19: {21: None, 15: None}, 3: {1:
            None, 7: None}, 13: {9: None, 15: None}}),

            Graph({128: {126: None, 4: None}, 126: {128: None, 130:
            None}, 2: {3: None, 4: None}, 130: {129: None, 126: None},
            131: {129: None}, 129: {130: None, 131: None}, 3: {2:
            None}, 4: {128: None, 2: None}}),
            ]

    def test_clustering(self):
        """this test is (obviously) a hack, but at least it will test when
        things change"""

        actual,nodes = checkerboard.points2graph(self.x,self.y,
                                                 aspect_ratio=self.aspect_ratio)
        assert actual==self.expected_graph

    def test_find_subgraph_similar_direction(self):
        graph,nodes = checkerboard.points2graph(self.x,self.y,
                                                aspect_ratio=self.aspect_ratio)
        #reduce number of comparisons -- some should be enough
        nodes = nodes[:3]
        similar_direction_graphs = []
        D2R = np.pi/180.0
        for node in nodes:
            subgraph = checkerboard.find_subgraph_similar_direction(
                graph,
                source=node,
                direction_eps_radians=10.0*D2R,
                already_done=similar_direction_graphs)
            if subgraph is not None:
                similar_direction_graphs.append( subgraph )

        # compare graphs
        assert len(self.expected_subgraphs)==len(similar_direction_graphs)
        for expected,actual in zip(self.expected_subgraphs,
                                   similar_direction_graphs):
            assert expected==actual
