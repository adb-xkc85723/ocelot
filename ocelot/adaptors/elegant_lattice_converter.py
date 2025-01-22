"""Elegant <--> Ocelot lattice converter"""

import re
import sys
import operator
import numpy as np
import yaml
import logging

from ocelot.cpbd.magnetic_lattice import *
from ocelot.cpbd.elements import *
from ocelot.cpbd.io import *
_logger = logging.getLogger(__name__)

class ElegantLatticeConverter:
    """Main Elegant <--> Ocelot lattice converter class"""

    def __init__(self):
        self.lsc_params = {}
        self.lsc_params.update({'LSC': 0})
        self.lsc_params.update({'BINS': 0})
        self.lsc_params.update({'LOW_FREQUENCY_CUTOFF0': -1})
        self.lsc_params.update({'LOW_FREQUENCY_CUTOFF1': -1})
        self.lsc_params.update({'HIGH_FREQUENCY_CUTOFF0': -1})
        self.lsc_params.update({'HIGH_FREQUENCY_CUTOFF1': -1})

        self.csr_params = {}
        # self.csr_params.update({'DRIFT': {}})
        self.csr_params.update({'CSRCSBEND': {}})
        # self.csr_params['DRIFT'].update({'USE_STUPAKOV': 1})
        self.csr_params['CSRCSBEND'].update({'CSR': 1})
        self.csr_params['CSRCSBEND'].update({'ISR': 1})
        self.csr_params['CSRCSBEND'].update({'N_KICKS': 20})
        self.csr_params['CSRCSBEND'].update({'EDGE1_EFFECTS': 1})
        self.csr_params['CSRCSBEND'].update({'EDGE2_EFFECTS': 1})
        self.csr_params['CSRCSBEND'].update({'EDGE_ORDER': 2})
        self.csr_params['CSRCSBEND'].update({'INTEGRATION_ORDER': 4})
        self.csr_params['CSRCSBEND'].update({'SG_HALFWIDTH': 1})
        self.csr_params['CSRCSBEND'].update({'SGDERIV_HALFWIDTH': 1})
        self.csr_params['CSRCSBEND'].update({'HIGH_FREQUENCY_CUTOFF0': -1})
        self.csr_params['CSRCSBEND'].update({'HIGH_FREQUENCY_CUTOFF1': -1})
        self.csr_params['CSRCSBEND'].update({'LOW_FREQUENCY_CUTOFF0': -1})
        self.csr_params['CSRCSBEND'].update({'LOW_FREQUENCY_CUTOFF1': -1})

        self.cavity_params = {}
        self.cavity_params.update({'CHANGE_P0': 1})
        self.cavity_params.update({'END1_FOCUS': 1})
        self.cavity_params.update({'END2_FOCUS': 1})
        self.cavity_params.update({'LSC': 0})
        self.cavity_params.update({'LSC_BINS': 0})
        self.cavity_params.update({'N_BINS': 100})
        self.cavity_params.update({'N_KICKS': 25})
        
        self.lh_und = 'UND_LH.01'
        self.lh_params = {}
        self.lh_params.update({'L': 0.04036 * 12})
        self.lh_params.update({'Bu': 0.2481})
        self.lh_params.update({'PERIODS': 12})
        self.lh_params.update({'METHOD': '\"non-adaptive runge-kutta\"'})
        self.lh_params.update({'FIELD_EXPANSION': 'ideal'})
        self.lh_params.update({'LASER_WAVELENGTH': 800e-9})
        self.lh_params.update({'LASER_PEAK_POWER': 1e5})
        self.lh_params.update({'LASER_W0': 150e-6})
        self.lh_params.update({'N_STEPS': 12*100})
        self.lh_params.update({'N_STEPS': 12 * 100})
        self.lh_params.update({'ACCURACY': 1e-6})

        self.twiss = {}

    def init_convert_matrix(self):
        """
        Init Elegant -> Ocelot convertion matrix
        in case of dict has list, e.g. 'VOLT':['v','1.0e-9'] it uses for translation different units e.g. eV to GeV
        in case, dict has list with len=3, e.g. 'R11': ['r', 0, 0] it uses as indices e.g. r[0, 0]
        """

        self.elegant_matrix = {}
        self.elegant_matrix['DRIF'] = {'type': Drift, 'params':{'L':'l'}}
        self.elegant_matrix['EDRIFT'] = {'type': Drift, 'params':{'L':'l'}}
        self.elegant_matrix['DRIFT'] = {'type': Drift, 'params': {'L': 'l'}}
        self.elegant_matrix['RFTM110'] = {'type': Drift, 'params': {'L': 'l'}}
        self.elegant_matrix['LSCDRIFT'] = {'type': Drift, 'params': {'L': 'l'}}
        # self.elegant_matrix['CSRDRIFT'] = {'type': Drift, 'params': {'L': 'l'}}
        self.elegant_matrix['SOLE'] = {'type': Solenoid, 'params': {'L': 'l'}}
        self.elegant_matrix['SBEN'] = {'type': SBend,
                                       'params':{'L': 'l', 'ANGLE': 'angle', "K1": "k1", "K2": "k2", "FINT": "fint",
                                                 'E1': 'e1', 'E2': 'e2', "HGAP": ["gap", "2"], 'TILT': 'tilt'}}
        self.elegant_matrix['SBEND'] = {'type': SBend,
                                        'params':{'L': 'l', 'ANGLE': 'angle', "K1": "k1", "K2": "k2", "FINT": "fint",
                                                 'E1': 'e1', 'E2': 'e2', "HGAP": ["gap", "2"], 'TILT': 'tilt'}}
        self.elegant_matrix['RBEN'] = {'type': RBend,
                                       'params':{'L': 'l', 'ANGLE': 'angle', "K1": "k1", "K2": "k2", "FINT": "fint",
                                                 'E1': 'e1', 'E2': 'e2', "HGAP": ["gap", "2"], 'TILT': 'tilt'}}
        self.elegant_matrix['QUAD'] = {'type': Quadrupole, 'params': {'L':'l', 'K1':'k1','TILT': 'tilt'}}
        self.elegant_matrix['KQUAD'] = {'type': Quadrupole, 'params': {'L':'l', 'K1':'k1','TILT': 'tilt'}}
        self.elegant_matrix['QUADRUPOLE'] = {'type': Quadrupole, 'params': {'L':'l', 'K1':'k1', 'TILT': 'tilt'}}
        self.elegant_matrix['SEXT'] = {'type': Sextupole, 'params': {'L':'l', 'K2':'k2'}}
        self.elegant_matrix['KSEXT'] = {'type': Sextupole, 'params': {'L':'l', 'K2':'k2'}}
        self.elegant_matrix['KOCT'] = {'type': Octupole, 'params': {'L':'l', 'K3':'k3'}}
        self.elegant_matrix['MONI'] = {'type': Monitor, 'params': {}}
        self.elegant_matrix['MONITOR'] = {'type': Monitor, 'params': {}}
        self.elegant_matrix['MARK'] = {'type': Marker, 'params': {}}
        # self.elegant_matrix['LASER'] = {'type': Laser, 'params': {}}
        # self.elegant_matrix['WATCH'] = {'type': Marker, 'params': {}}
        self.elegant_matrix['RFCA'] = {'type': Cavity, 'params': {'L':'l', 'VOLT':['v','1.0e-9'], 'FREQ': 'freq', 'PHASE': 'phi'}}
        self.elegant_matrix['HKICK'] = {'type': Hcor, 'params': {'L':'l'}}
        self.elegant_matrix['VKICK'] = {'type': Vcor, 'params': {'L':'l'}}
        self.elegant_matrix['KICKER'] = {'type': Vcor, 'params': {'L':'l'}}
        self.elegant_matrix['WIGGLER'] = {'type': Undulator, 'params': {'L': 'l', 'K': 'Ky', "POLES": ["nperiods", "0.5"]}}
        self.elegant_matrix['EMATRIX'] = {'type': Matrix, 'params': {'L':'l',
                                                'R11': ['r', 0, 0], 'R12': ['r', 0, 1], 'R13': ['r', 0, 2], 'R14': ['r', 0, 3], 'R15': ['r', 0, 4], 'R16': ['r', 0, 5],
                                                'R21': ['r', 1, 0], 'R22': ['r', 1, 1], 'R23': ['r', 1, 2], 'R24': ['r', 1, 3], 'R25': ['r', 1, 4], 'R26': ['r', 1, 5],
                                                'R31': ['r', 2, 0], 'R32': ['r', 2, 1], 'R33': ['r', 2, 2], 'R34': ['r', 2, 3], 'R35': ['r', 2, 4], 'R36': ['r', 2, 5],
                                                'R41': ['r', 3, 0], 'R42': ['r', 3, 1], 'R43': ['r', 3, 2], 'R44': ['r', 3, 3], 'R45': ['r', 3, 4], 'R46': ['r', 3, 5],
                                                'R51': ['r', 4, 0], 'R52': ['r', 4, 1], 'R53': ['r', 4, 2], 'R54': ['r', 4, 3], 'R55': ['r', 4, 4], 'R56': ['r', 4, 5],
                                                'R61': ['r', 5, 0], 'R62': ['r', 5, 1], 'R63': ['r', 5, 2], 'R64': ['r', 5, 3], 'R65': ['r', 5, 4], 'R66': ['r', 5, 5],}}
        self.elegant_matrix['CSRCSBEND'] = {'type': SBend, 'params': {'L':'l', 'ANGLE': 'angle', "K1": "k1", "K2": "k2", 'E1': 'e1', 'E2': 'e2', 'TILT': 'tilt'}}
        self.elegant_matrix['CSRCSBEN'] = {'type': RBend, 'params': {'L':'l', 'ANGLE': 'angle', "K1": "k1", "K2": "k2", 'E1': 'e1', 'E2': 'e2', 'TILT': 'tilt'}}
        self.elegant_matrix['RFCW'] = {'type': Cavity, 'params': {'L': 'l', 'VOLT': ['v','1.0e-9'], 'FREQ': 'freq', 'PHASE': 'phi'}}

    def fix_convert_matrix(self):
        """Init and fix Ocelot -> Elegant convertion matrix"""

        self.init_convert_matrix()

        # Some elements are deleted from the matrix to exclude ambiguity in elements conversion
#         del self.elegant_matrix['CSRCSBEND']
#         del self.elegant_matrix['RFCW']


    def calc_rpn(self, expression, constants={}, info=''):
        """
        Calculation of the expression written in reversed polish notation
        """

        operators = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
        functions = {'SIN': np.sin, 'COS': np.cos, 'TAN': np.tan, 'ASIN': np.arcsin, 'ACOS': np.arccos, 'ATAN': np.arctan, 'SQRT': np.sqrt}

        stack = [0]
        for token in expression.split(" "):

            if token in operators:
                op2, op1 = stack.pop(), stack.pop()
                stack.append(operators[token](op1, op2))

            elif token in constants:
                stack.append(constants[token])

            elif token in functions:
                op = stack.pop()
                stack.append(functions[token](op))

            elif token:
                try:
                    stack.append(float(token))
                except ValueError:
                    print('********* ERROR! NO ' + token + ' in function conversion or constants list in calc_rpn function. ' + info)
                    sys.exit()

        return stack.pop()


    def convert_val(self, str, constants, info):
        """
        Convert string input value to float or change input string value by value from constants array
        """

        try:
            value = float(str)
        except ValueError:
            if str in constants:
                value = constants[str]
            else:
                value = self.calc_rpn(str, constants, 'In element '+info)

        return value


    def elegant2ocelot(self, file_name, multiple_cells=False):
        """
        :filename - input Elegant lattice filename
        """

        # just in case double check
        # because in self.ocelot2elegant function these elements are deleted
        self.init_convert_matrix()

        with open(file_name) as file_link:
            data = file_link.read()

        # delete comments
        data = re.sub(r'![^\n]*\n', '\n', data)
        # merge splitted lines
        data = re.sub(r'&\s*\n', '', data)

        # element names correction
        # remove parentheses from the names
        bad_element_names = re.findall(r'.\[.*\].*:', data)
        for element in bad_element_names:
            element = element.replace(":", "")
            element = element.strip()
            element_new = element.replace("[", "")
            element_new = element_new.replace("]", "")
            data = data.replace(element, element_new)

        # element names correction
        bad_element_names = re.findall(r'"(.*)":', data)
        for element in bad_element_names:
            element_new = re.sub(r'[\W]+', '_', element)
            data = re.sub(r''+str(element), str(element_new), data)

        #data = re.sub(r'"', '', data)
        data = re.sub(r'\'', '"', data)
        lines = data.split('\n')

        # parsing lines
        elegant_elements_dict = {}
        used_cell_name = ''
        cell_flag = False
        constants = {}
        elegant_cells = []
        used_cell_names = []
        used_line_names = []
        all_cell_names = []
        for line in lines:
            # print(line)
            string = line.upper().strip()

            # skip empty and comment lines
            if string == '' or string[0] == '!':
                continue

            # parsing constants
            if string[0] == '%' and string.find('STO') != -1:
                expr = string[1:].split('STO')
                constants[expr[1].strip()] = self.calc_rpn(expr[0].strip(), constants, 'In string '+string)
                continue

            # delete spaces and quotes
            string0 = string
            string = ''
            del_space = True
            for s in string0:
                if s == '"':
                    del_space = not del_space
                    continue
                elif s == ' ' and del_space:
                    continue
                string += s

            # split line and add to the dictionary with element names and descriptions
            element_desc = string.split(':')
            element_params = ''

            for i in range(1, len(element_desc)):
                element_params += '_' + str(element_desc[i])
            element_params = element_params[1:]
            elegant_elements_dict.update({element_desc[0]:element_params})

            # finding used lattice cell
            if cell_flag is False and element_params != '':
                used_cell_name = element_desc[0]
                used_cell_names.append(element_desc[0])
            if cell_flag is False and element_desc[0][:3] == 'USE':
                used_cell_name = element_desc[0][4:]
                used_cell_names.append(element_desc[0][4:])
                cell_flag = True

        # parsing found used cell
        if used_cell_name in elegant_elements_dict:
            elegant_cell = elegant_elements_dict[used_cell_name]
        else:
            print('********* ERROR! used unkown cell or element ' + used_cell_name + ' *********')
            sys.exit()
        for used_cell in used_cell_names:
            if used_cell in elegant_elements_dict:
                elegant_cells.append(elegant_elements_dict[used_cell])
            else:
                print('********* ERROR! used unkown cell or element ' + used_cell_name + ' *********')
                sys.exit()
        # replace multiplications
        elegant_cell = self.replace_s_multiplications(elegant_cell)
        elegant_cell = self.replace_multiplications(elegant_cell)
        # replace cells by elements
        elegant_cell = self.replace_cells(elegant_cell, elegant_elements_dict)
        full_cells = []
        element_cell_names = []
        indices = []
        i = 0
        for cell in elegant_cells:
            # replace multiplications
            cell = self.replace_s_multiplications(cell)
            cell = self.replace_multiplications(cell)
            # replace cells by elements
            cell = self.replace_cells(cell, elegant_elements_dict)
            if len(cell) > 1:
                full_cells.append(cell)
                element_cell_names.append(used_cell_names[i])
                indices.append(i)
            i += 1
        # print(elegant_cell)
        # create Ocelot cell

        if multiple_cells:
            ocelot_cells = []
            for cell in full_cells:
                ocelot_cell = ()
                elements_list = {}
                for elem in cell:
                    if elem not in elements_list.keys():

                        param = elegant_elements_dict[elem].split(',')
                        param = [elem.strip() for elem in param]
                        # convert element
                        if param[0] in self.elegant_matrix.keys():
                            # create element
                            elements_list[elem] = self.elegant_matrix[param[0]]['type'](eid=elem)

                            # parse parameters
                            for data in param:
                                result = data.split('=')
                                if result[0] in self.elegant_matrix[param[0]]['params']:
                                    tmp = self.elegant_matrix[param[0]]['params'][result[0]]

                                    val = self.convert_val(result[1], constants, elem)
                                    if tmp.__class__ == list:
                                        if len(tmp) == 2:
                                            elements_list[elem].__dict__[tmp[0]] = val * float(tmp[1])
                                            setattr(elements_list[elem], tmp[0], val * float(tmp[1]))
                                        elif len(tmp) == 3:
                                            atnam = getattr(elements_list[elem], tmp[0])
                                            atnam[tmp[1]][tmp[2]] = val * float(tmp[1])
                                            #elements_list[elem].set_element(tmp[0], tmp[1], tmp[2], val * float(tmp[1]))
                                            #setattr(elements_list[elem], [tmp[0]][tmp[1], tmp[2]], val * float(tmp[1]))
                                    else:
                                        # fix for phi cavity
                                        if elements_list[elem].__class__ == Cavity and tmp == 'phi':
                                            val = 90.0 - val
                                        elements_list[elem].__dict__[tmp] = val
                                        setattr(elements_list[elem], tmp, val)
                            if elements_list[elem].__class__ == Undulator:
                                elements_list[elem].lperiod = elements_list[elem].l/elements_list[elem].nperiods
                        # replace element by Drift (if it has L) or skip
                        else:
                            elements_list[elem] = None
                            print_skiped = True

                            for data in param:
                                if data[:2] == 'L=':

                                    val = self.convert_val(data[2:], constants, elem)

                                    elements_list[elem] = Drift(eid=elem, l=val)
                                    print_skiped = False
                                    break
                            params = ['RCOLLIMATOR', 'CHARGE', 'SCATTER', 'TWISS']
                            if print_skiped:
                                if not param[0] in params:
                                    print('WARNING! Unknown element', elem, 'with type', param[0], 'was skipped')
                            else:
                                if not param[0] in params:
                                    print('WARNING! Unknown element', elem, 'with type', param[0], 'was changed by Drift')


                    if elements_list[elem] is not None:
                        ocelot_cell += (elements_list[elem],)
                ocelot_cells.append(ocelot_cell)
            return [ocelot_cells, element_cell_names]
        else:
            ocelot_cell = ()
            elements_list = {}
            for elem in elegant_cell:

                if elem not in elements_list.keys():

                    param = elegant_elements_dict[elem].split(',')
                    param = [elem.strip() for elem in param]
                    # convert element
                    if param[0] in self.elegant_matrix.keys():
                        # create element
                        elements_list[elem] = self.elegant_matrix[param[0]]['type'](eid=elem)

                        # parse parameters
                        for data in param:
                            result = data.split('=')
                            if result[0] in self.elegant_matrix[param[0]]['params']:
                                tmp = self.elegant_matrix[param[0]]['params'][result[0]]

                                val = self.convert_val(result[1], constants, elem)

                                if tmp.__class__ == list:
                                    if len(tmp) == 2:
                                        elements_list[elem].__dict__[tmp[0]] = val * float(tmp[1])
                                    elif len(tmp) == 3:
                                        print(elements_list[elem], tmp)
                                        elements_list[elem].__dict__[tmp[0]][tmp[1], tmp[2]] = val
                                else:
                                    # fix for phi cavity
                                    if elements_list[elem].__class__ == Cavity and tmp == 'phi':
                                        val = 90.0 - val
                                    elements_list[elem].__dict__[tmp] = val
                        if elements_list[elem].__class__ == Undulator:
                            elements_list[elem].lperiod = elements_list[elem].l/elements_list[elem].nperiods
                    # replace element by Drift (if it has L) or skip
                    else:
                        elements_list[elem] = None
                        print_skiped = True

                        for data in param:
                            if data[:2] == 'L=':

                                val = self.convert_val(data[2:], constants, elem)

                                elements_list[elem] = Drift(eid=elem, l=val)
                                print_skiped = False
                                break

                        if print_skiped:
                            print('WARNING! Unknown element', elem, 'with type', param[0], 'was skipped')
                        else:
                            print('WARNING! Unknown element', elem, 'with type', param[0], 'was changed by Drift')


                if elements_list[elem] is not None:
                    ocelot_cell += (elements_list[elem],)

            return ocelot_cell


    def replace_s_multiplications(self, line):
        """Replace single multiplication of elements"""

        rep_data = re.findall(r'(\d+)\*([\w]+)', line)

        for arr in rep_data:
            old_element = arr[0]+'*'+arr[1]
            new_element = arr[1] + (',' + arr[1]) * (int(arr[0]) - 1)
            line = line.replace(old_element, new_element)

        return line


    def replace_multiplications(self, line):
        """Replace multiplication with brackets of elements"""

        rep_data = re.findall(r'(\d+)\*\(([\w\,]+)\)', line)

        while rep_data != []:

            for arr in rep_data:
                old_element = arr[0]+'*('+arr[1]+')'
                new_element = arr[1] + (',' + arr[1]) * (int(arr[0]) - 1)
                line = line.replace(old_element, new_element)

            rep_data = re.findall(r'(\d+)\*\(([\w\,]+)\)', line)

        return line


    def replace_cells(self, cell, elements_dict):
        """Replace cells by elements"""

        if cell[0:4] != 'LINE':
            return [cell]

        cell = cell[6:-1]

        elements_list = cell.split(',')
        elements_list = [elem.strip() for elem in elements_list]
        check_lines = True
        while check_lines:
            new_list = []
            check_lines = False

            for i in range(0, len(elements_list)):
                reverse = False
                if elements_list[i][0] == '-':
                    reverse = True
                    elements_list[i] = elements_list[i][1:]

                if elements_list[i] not in elements_dict:
                    print('********* ERROR! used undefined cell or element ' + elements_list[i] + ' *********')
                    sys.exit()

                if elements_dict[elements_list[i]][0:4] == 'LINE':
                    check_lines = True
                    tmp_list = elements_dict[elements_list[i]][6:-1].split(',')
                    if reverse:
                        start, stop, step = len(tmp_list)-1, -1, -1
                    else:
                        start, stop, step = 0, len(tmp_list), 1
                    for j in range(start, stop, step):
                        new_list.append(tmp_list[j])
                else:
                    new_list.append(elements_list[i])

            elements_list = new_list

        return elements_list


    def ocelot2elegant(self, lattice, file_name='lattice.lte', charge=None):
        """
        :lattice - Ocelot lattice objectl
        :filename - output Elegant lattice filename
        """

        # fix due to elements type doubling
        self.fix_convert_matrix()

        reverse_matrix = {}
        elements_arr = {}
        elegant_cell = []

        for elegant_type in self.elegant_matrix:
            if elegant_type == 'LASER':
                elegant_type = 'MONITOR'
            reverse_matrix.update({self.elegant_matrix[elegant_type]['type'].__name__:elegant_type})
            elements_arr.update({self.elegant_matrix[elegant_type]['type'].__name__:[]})

        if charge is not None:
            elegant_cell.append('CH')

        if self.twiss:
            elegant_cell.append('TWISS')

        for elem in lattice.sequence:

            # add element if its class it the matrix
            if elem.__class__.__name__ in reverse_matrix.keys():

                if elem not in elements_arr[elem.__class__.__name__]:
                    elements_arr[elem.__class__.__name__].append(elem)

            # if element class not in the maxrix - replace it by Drift (L > 0.0) or skip it
            else:
                if elem.l > 0.0:
                    elements_arr['Drift'].append(Drift(l=elem.l, eid=elem.id))
                else:
                    continue

            # add element in the final cell
            elegant_cell.append(elem.id)

        #  save data to file
        lines = ''

        if charge is not None:
            lines += f'CH:CHARGE,total={charge}\n'

        if self.twiss:
            twsstr = 'TWISS:TWISS_INPUT,'
            for k, v in self.twiss.items():
                twsstr += f'{k}={v},'
            twsstr += 'FROM_BEAM=1'
            lines += twsstr + '\n'

        # save elements
        for element_class in elements_arr:
            self.linelen = 0
            for elem in elements_arr[element_class]:
                if elem.id == self.lh_und:
                    lines += elem.id + ': ' + 'LSRMDLTR'
                else:
                    lines += elem.id + ': ' + reverse_matrix[element_class]

                # check parameter 'L' and print it first
                    if 'L' in self.elegant_matrix[reverse_matrix[element_class]]['params']:
                        tmp = self.elegant_matrix[reverse_matrix[element_class]]['params']['L']
                        lines += ',L=' + str(elem.l)
    
                    # print other parameters
                    for param in self.elegant_matrix[reverse_matrix[element_class]]['params']:
                        self.newlin = self.get_newline(lines)
                        if param == 'L':
                            continue
                        tmp = self.elegant_matrix[reverse_matrix[element_class]]['params'][param]
                        if tmp.__class__ == list:
                            if len(tmp) == 2:
                                lines += ',' + self.newlin + param + '=' + str(getattr(elem, tmp[0])/float(tmp[1]))
    #                             lines += ',' + param + '=' + str(elem.__dict__[tmp[0]]/float(tmp[1]))
                            elif len(tmp) == 3:
                                lines += ',' + self.newlin + param + '=' + str(getattr(elem, tmp[0])[tmp[1], tmp[2]])
                        else:
                            # fix for phi cavity
                            if elem.__class__ == Cavity and tmp == 'phi':
                                lines += ',' + self.newlin + param + '=' + str(90.0 - getattr(elem, tmp))
                                lines += ',CHANGE_P0=1,END1_FOCUS=1,END2_FOCUS=1,BODY_FOCUS_MODEL="SRS"'
                            elif elem.__class__ == Matrix and "rm" in tmp:
                                i, j = int(int(param[-2]) - 1), int(int(param[-1]) - 1)
                                lines += ',' + self.newlin + param + '=' + str(getattr(elem, "r")[i, j])
                            else:
    #                             print(elem.__dict__)
                                lines += ',' + self.newlin + param + '=' + str(getattr(elem, tmp))
    #                             lines += ',' + param + '=' + str(elem.__dict__[tmp])

                if elem.__class__ == Drift:
                    for k, v in self.lsc_params.items():
                        self.newlin = self.get_newline(lines)
                        if isinstance(v, dict):
                            if len(list(v.keys())) > 0:
                                lines += ',' + self.newlin + k + '=' + str(v)
                        else:
                            lines += ',' + self.newlin + k + '=' + str(v)
                    for k, v in self.lsc_params.items():
                        if k != 'LSC':
                            self.newlin = self.get_newline(lines)
                            if isinstance(v, dict):
                                if len(list(v.keys())) > 0:
                                    lines += ',' + self.newlin + k + '=' + str(v)
                            else:
                                lines += ',' + self.newlin + k + '=' + str(v)
                if elem.__class__ in [SBend, RBend]:
                    for k, v in self.csr_params['CSRCSBEND'].items():
                        self.newlin = self.get_newline(lines)
                        if isinstance(v, dict):
                            if len(list(v.keys())) > 0:
                                lines += ',' + self.newlin + k + '=' + str(v)
                        else:
                            lines += ',' + self.newlin + k + '=' + str(v)
                if elem.__class__ == Cavity:
                    for k, v in self.cavity_params.items():
                        self.newlin = self.get_newline(lines)
                        if isinstance(v, dict):
                            if (len(list(v.keys())) > 0) and (k == elem.id):
                                for subk, subv in v.items():
                                    lines += ',' + self.newlin + subk + '=' + str(subv)
                        else:
                            lines += ',' + self.newlin + k + '=' + str(v)
                if elem.id == self.lh_und:
                    for k, v in self.lh_params.items():
                        self.newlin = self.get_newline(lines)
                        if isinstance(v, dict):
                            if (len(list(v.keys())) > 0) and (k == elem.id):
                                for subk, subv in v.items():
                                    lines += ',' + self.newlin + subk + '=' + str(subv)
                        else:
                            lines += ',' + self.newlin + k + '=' + str(v)            
                
                lines += '\n'

        # save cell
        lines += 'CELL: LINE=(' + ', '.join(elegant_cell) + ')\n'

#         linesplit = lines.split('\n')
#         linesnew = ''
#         for line in linesplit:
#             if len(line) > 80:
#                 linecommasplit = line.split(',')
#                 for i, l in enumerate(linecommasplit):
#                     if len(''.join(linecommasplit[0: i])) > 80:
#                         linecommasplit[i] += ', &\n'
#                         break
#                 linesnew += ', '.join(linecommasplit).replace(', &\n,', ', &\n') + '\n'
#             else:
#                 linesnew += line + '\n'

#         lines = linesnew

        lines += 'USE,CELL\n'
        lines += 'RETURN\n'

        fff = open(file_name, 'w')
        fff.writelines(lines)
        fff.close()

        return 0

    def get_newline(self, lines):
        self.linelen = len(lines.split('\n')[-1])
        if self.linelen > 80:
            return ' &\n'
        else:
            return ''

class ElegantInputGenerator:

    def __init__(self, kwlocation):
        self.global_settings_required = ['mpi_io_write_buffer_size']
        self.run_setup_required = ['lattice', 'rootname', 'use_beamline']
        self.twiss_output_required = ['filename']
        self.sdds_beam_required = ['input']
        self.bunched_beam_required = ['bunch']
        self.matrix_output_required = ['SDDS_output']
        with open(kwlocation, 'r') as stream:
            try:
                self.ele_keywords = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
    def global_settings(self, **kwargs):
        if not all(elem in kwargs for elem in self.global_settings_required):
            _logger.warning(f"ERROR!!!! Missing values in inputs to run_setup: {(set(self.global_settings_required).difference(kwargs))}")
            _logger.warning("Please consult the ELEGANT manual")
            return '0'
        self.inp = '&global_settings    \n'
        for k, v in kwargs.items():
            if k not in self.ele_keywords['global_settings']:
                _logger.warning(f"ERROR!!!! Command {k} not allowed in &global_settings")
                _logger.warning("Please consult the ELEGANT manual")
                return '0'
            self.inp += f'    {k} = {v} \n'
        self.inp += '&end    \n\n'
        return self.inp

    def run_setup(self, **kwargs):
        if not all(elem in kwargs for elem in self.run_setup_required):
            _logger.warning(f"ERROR!!!! Missing values in inputs to run_setup: {(set(self.run_setup_required).difference(kwargs))}")
            _logger.warning("Please consult the ELEGANT manual")
            return '0'
        self.inp = '&run_setup    \n'
        for k, v in kwargs.items():
            if k not in self.ele_keywords['run_setup']:
                _logger.warning(f"ERROR!!!! Command {k} not allowed in &run_setup")
                _logger.warning("Please consult the ELEGANT manual")
                return '0'
            self.inp += f'    {k} = {v} \n'
        self.inp += '&end    \n\n'
        return self.inp

    def run_control(self, **kwargs):
        self.inp = '&run_control    \n'
        for k, v in kwargs.items():
            if k not in self.ele_keywords['run_control']:
                _logger.warning(f"ERROR!!!! Command {k} not allowed in &run_control")
                _logger.warning("Please consult the ELEGANT manual")
                return '0'
            self.inp += f'    {k} = {v} \n'
        self.inp += '&end    \n\n'
        return self.inp

    def twiss_output(self, **kwargs):
        self.inp = '&twiss_output    \n'
        if 'matched' in list(kwargs.keys()):
            self.mat = int(kwargs['matched'])
            if self.mat == 1:
                self.matched = True
            else:
                self.matched = False
        else:
            self.matched = False
        if not self.matched:
            self.twiss_output_required = ['filename', 'beta_x', 'beta_y', 'alpha_x', 'alpha_y']
        else:
            self.twiss_output_required = ['filename']
        if not all(elem in kwargs for elem in self.twiss_output_required):
            _logger.warning(f"ERROR!!!! Missing values in inputs to twiss_output: {(set(self.twiss_output_required).difference(kwargs))}")
            _logger.warning("Please consult the ELEGANT manual")
            return '0'
        for k, v in kwargs.items():
            if k not in self.ele_keywords['twiss_output']:
                _logger.warning(f"ERROR!!!! Command {k} not allowed in &twiss_output")
                _logger.warning("Please consult the ELEGANT manual")
                return '0'
            self.inp += f'    {k} = {v} \n'
        self.inp += '&end    \n\n'
        return self.inp

    def sdds_beam(self, **kwargs):
        if not all(elem in kwargs for elem in self.sdds_beam_required):
            _logger.warning(f"ERROR!!!! Missing values in inputs to sdds_beam: {(set(self.sdds_beam_required).difference(kwargs))}")
            _logger.warning("Please consult the ELEGANT manual")
            return '0'
        self.inp = '&sdds_beam    \n'
        for k, v in kwargs.items():
            if k not in self.ele_keywords['sdds_beam']:
                _logger.warning(f"ERROR!!!! Command {k} not allowed in &sdds_beam")
                _logger.warning("Please consult the ELEGANT manual")
                return '0'
            self.inp += f'    {k} = {v} \n'
        self.inp += '&end    \n\n'
        return self.inp

    def bunched_beam(self, **kwargs):
        if not all(elem in kwargs for elem in self.bunched_beam_required):
            _logger.warning(f"ERROR!!!! Missing values in inputs to bunched_beam: {(set(self.bunched_beam_required).difference(kwargs))}")
            _logger.warning("Please consult the ELEGANT manual")
            return '0'
        self.inp = '&bunched_beam    \n'
        for k, v in kwargs.items():
            if k not in self.ele_keywords['bunched_beam']:
                _logger.warning(
                    f"ERROR!!!! Command {k} not allowed in &bunched_beam")
                _logger.warning("Please consult the ELEGANT manual")
                return '0'
            self.inp += f'    {k} = {v} \n'
        self.inp += '&end    \n\n'
        return self.inp

    def matrix_output(self, **kwargs):
        if not all(elem in kwargs for elem in self.matrix_output_required):
            _logger.warning(f"ERROR!!!! Missing values in inputs to matrix_output: {(set(self.matrix_output_required).difference(kwargs))}")
            _logger.warning("Please consult the ELEGANT manual")
            return '0'
        self.inp = '&matrix_output    \n'
        for k, v in kwargs.items():
            if k not in self.ele_keywords['matrix_output']:
                _logger.warning(f"ERROR!!!! Command {k} not allowed in &matrix_output")
                _logger.warning("Please consult the ELEGANT manual")
                return '0'
            self.inp += f'    {k} = {v} \n'
        self.inp += '&end    \n\n'
        return self.inp

    def track(self, **kwargs):
        self.inp = '&track    \n'
        self.inp += '&end    \n\n'
        return self.inp

    def stop(self, **kwargs):
        self.inp = '&stop    \n'
        self.inp += '&end    \n\n'
        return self.inp

    def save_elegant_input(self, input, filename):
        with open(filename, 'w') as f:
            f.write(input)

if __name__ == '__main__':
    pass
    
    """
    # example of Elegant - Ocelot convertion
    from ocelot.adaptors.elegant_lattice_converter import *
    SC = ElegantLatticeConverter()
    read_cell = SC.elegant2ocelot('elbe.lte')
    lattice = MagneticLattice(read_cell)
    write_lattice(lattice, file_name="lattice.py", remove_rep_drifts=False)
    """
    
    """
    # example of Ocelot - Elegant convertion
    from lattice import *
    lattice = MagneticLattice(cell)
    SC = ElegantLatticeConverter()
    SC.ocelot2elegant(lattice)
    """
