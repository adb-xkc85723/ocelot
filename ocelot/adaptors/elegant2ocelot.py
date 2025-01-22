'''
Created on 05.04.2013
@author: I. Zagororodnov
modified by S. Tomin

Usage:
# file.flo and file.par must be converted in text format (e.g. by sdds2spreadsheet.exe)
sequence = read_lattice_elegant(file_flo=file_flo.txt, file_par=file_par.txt)
lat = MagneticLattice(sequence)
'''
import csv
from ocelot.cpbd.elements import *
from ocelot.cpbd.beam import *
from math import *
import sys
import numpy as np
import subprocess
import ASTeCsdds.sdds as sdds

def read_file(filename):
    if sys.version_info[0] < 3:
        f=open(filename, 'rb')
    else:
        f = open(filename, 'r', newline='', encoding='utf8')
    data=csv.reader(f, delimiter='\t')
    data=[row for row in data]
    f.close()
    return data

def elegantBeam2particleArray(filename, charge=None):
    subprocess.run(['sddsconvert -ascii ' + filename], shell=True)
    sddsf = sdds.SDDS(0)
    sddsf.load(filename)
    fc = open(filename, "r").readlines()
    sddsdata = {}
    sddsdata.update({'parameter': {}})
    sddsdata.update({'column': {}})
    paramindex = 0
    columnindex = 0
    index = 0
    for n, d in zip(sddsf.parameterName, sddsf.parameterData):
        sddsdata['parameter'].update({n: d[0]})
    for n, d in zip(sddsf.columnName, sddsf.columnData):
        sddsdata['column'].update({n: d[0]})
    n = int(sddsdata['parameter']['Particles'])
    # allcols = np.zeros((n, len(list(sddsdata['column'].keys()))))
    # for i, f in enumerate(params):
    #     pkey = list(sddsdata['parameter'].keys())[i]
    #     if sddsdata['parameter'][pkey]['type'] == 'string':
    #         sddsdata['parameter'][pkey].update({'value': f.strip("\n")})
    #     else:
    #         sddsdata['parameter'][pkey].update({'value': float(f.strip("\n"))})
    # for i, f in enumerate(cols):
    #     allcols[i] = [float(o) for o in f.split()]
    # for i, c in enumerate(np.transpose(allcols)):
    #     ckey = list(sddsdata['column'].keys())[i]
    #     sddsdata['column'][ckey].update({'value': c})
    p_array = ParticleArray(n=n)
    try:
        p_array.E = sddsdata['parameter']['pCentral'] * 0.511 * 1e-3
        pCentral = sddsdata['parameter']['pCentral']
    except:
        p_array.E = np.mean(sddsdata['column']['p']) * 0.511 * 1e-3
        pCentral = np.mean(sddsdata['column']['p'])

    p = sddsdata['column']['p']
    E = [pp * 0.511 * 1e-3 for pp in p]
    p_array.rparticles[5] = [((e - (pCentral * 0.511 * 1e-3)) / p_array.E) for e in E]

    p_array.rparticles[0] = sddsdata['column']['x']
    p_array.rparticles[1] = sddsdata['column']['xp']
    p_array.rparticles[2] = sddsdata['column']['y']
    p_array.rparticles[3] = sddsdata['column']['yp']
    p_array.rparticles[4] = [ti * 299792458 for ti in sddsdata['column']['t']]
    try:
        p_array.q_array = np.ones(n) * sddsdata['parameter']['Charge'] / n
    except:
        p_array.q_array = np.ones(n) * charge / n

    try:
        p_array.s = sddsdata['parameter']['s']
    except:
        p_array.s = np.mean(p_array.rparticles[4])

    return p_array


def particleArray2elegantBeam(p_array, filename):
    p=p_array.energies*1e3/0.511
    t=p_array.tau()/299792458
    np.savetxt('test.dat',np.transpose([p_array.x(),p_array.px(),p_array.y(),p_array.py(),t,p]))
    with open('test.dat','r') as f:
        c=f.readlines()

    c.insert(0,str(p_array.E*1e3/0.511)+'\n')
    c.insert(1,str(np.sum(p_array.q_array))+'\n')
    c.insert(2,str(p_array.s))
    with open('test.dat','w') as f:
        c="".join(c)
        f.write(c)
    subprocess.run(['plaindata2sdds test.dat ' + filename + ' -separator=" " -noRowCount -parameter=pCentral,float,units="m\$be\$nc" -parameter=Charge,float,units=C -parameter=s,float,units=m -column=x,double,units=m -column=xp,double -column=y,double,units=m -column=yp,double -column=t,double,units=s -column=p,double,units="m\$be$nc"'],shell=True)


def read_twi_file(namefile):
    data = read_file(namefile)
    S = []
    Bx = []
    By = []
    for d in data[59:]:
        S.append(float(d[0]))
        Bx.append(float(d[1]))
        By.append(float(d[7]))
    return S, Bx, By


def read_lattice_elegant(file_flo, file_par):

    data_flo = read_file(file_flo)
    data_par = read_file(file_par)

    lattice=[]
    n_flo=len(data_flo)
    flo_params = np.array(data_flo[5])
    #print flo_params, np.where(flo_params == "ElementType")[0][0]
    i_s = np.where(flo_params == "s")[0][0] # i_X=1; i_Y=2;
    i_Z = np.where(flo_params == "Z")[0][0] #i_theta=4;i_phi=5;    i_psi=6;
    i_ElementName = np.where(flo_params == "ElementName")[0][0] # i_ElementOccurence=8;
    i_ElementType = np.where(flo_params == "ElementType")[0][0]
    for i in range(8, n_flo):
        v=data_flo[i] 
        sname=v[i_ElementName]
        stype=v[i_ElementType]
        sname=sname.replace('-C','')
        #sname=sname.replace('[','_')
        #sname=sname.replace(']','')
        #sname=sname.replace('-','_')
        #sname=sname.replace('.','_')
        print( stype,sname)
        if stype=='QUAD':
            quad = Quadrupole(eid=sname)
            quad.s=eval(v[i_s])
            quad.z=eval(v[i_Z])
            lattice=lattice+[quad]
        elif stype in ["DRIF", "LSCDRIFT", "CSRDRIFT", "KICKER", "ECOL"]:
            drift = Drift(eid=sname)
            drift.s=eval(v[i_s])
            drift.z=eval(v[i_Z])
            lattice=lattice+[drift]
        elif stype in ["MARK", "WATCH"]:
            mark = Marker(eid=sname)
            mark.s=eval(v[i_s])
            mark.z=eval(v[i_Z])
            lattice=lattice+[mark]
        elif stype=='SEXT':
            sext = Sextupole(eid=sname)
            sext.s=eval(v[i_s])
            sext.z=eval(v[i_Z])
            lattice=lattice+[sext]
        elif (stype=='CSBEND') or (stype == 'CSRCSBEND') or (stype == 'SBEN'):
            sben = Bend(l=1, eid=sname)
            sben.s=eval(v[i_s])
            sben.z=eval(v[i_Z])
            lattice=lattice+[sben]
        elif (stype=='CRBEND') or (stype=='CSRCRBEND'):
            rben= RBend (eid=sname)
            rben.s=eval(v[i_s])
            rben.z=eval(v[i_Z])
            lattice=lattice+[rben]
        elif stype=='WIGGLER':
            undulator = Undulator(eid=sname, lperiod=0, nperiods=0, Kx=0)
            undulator.s=eval(v[i_s])
            undulator.z=eval(v[i_Z])
            lattice=lattice+[undulator]   
        elif stype=='RFCA':
            cavity = Cavity(eid=sname, l=0)
            cavity.s=eval(v[i_s])
            cavity.z=eval(v[i_Z])
            lattice=lattice+[cavity]   
        elif stype=='HKICK':
            hcor = Hcor(eid=sname)
            hcor.s=eval(v[i_s])
            hcor.z=eval(v[i_Z])
            lattice=lattice+[hcor]
        elif stype=='VKICK':
            vcor = Vcor(eid=sname)
            vcor.s=eval(v[i_s])
            vcor.z=eval(v[i_Z])
            lattice=lattice+[vcor]
        elif stype=='MONI':
            monitor = Monitor(eid=sname)
            monitor.s=eval(v[i_s])
            monitor.z=eval(v[i_Z])
            lattice=lattice+[monitor]
            
    n_par=len(data_par)
    pos=6
    for elem in lattice:
        name0=elem.id
        name1=data_par[pos][0]
        while (pos<n_par-1) and (name0!=name1):
            pos=pos+1
            name1=data_par[pos][0]
            stype=data_par[pos][3]
        if pos>n_par-2:
            break
        elem.l=eval(data_par[pos][2])
        if elem.__class__ == Quadrupole:
            elem.k1=eval(data_par[pos+1][2])
            elem.tilt = eval(data_par[pos+2][2])
        elif elem.__class__ == Sextupole:
            elem.k2=eval(data_par[pos+1][2])
        elif elem.__class__ == Bend:
            elem.angle=eval(data_par[pos+1][2])
            if stype in ['CRBEND', 'CSRCRBEND', 'CSRCSBEND', "CSBEND"]:
                elem.e1 = eval(data_par[pos+10][2])
                elem.e2 = eval(data_par[pos+11][2])
                elem.tilt = eval(data_par[pos+12][2])
                elem.fint1 = eval(data_par[pos+16][2])
                elem.fint2 = eval(data_par[pos+16][2])
            if stype=='SBEN':
                elem.e1=eval(data_par[pos+3][2])
                elem.e2=eval(data_par[pos+4][2])
                elem.tilt=eval(data_par[pos+5][2])
        elif elem.__class__ == RBend:
            elem.angle=eval(data_par[pos+1][2])
            elem.tilt=eval(data_par[pos+12][2])
        elif elem.__class__ == Undulator:
            elem.l=eval(data_par[pos][2])
            #print( elem.l)
            elem.nperiods=eval(data_par[pos+8][2])/2
            #print (elem.nperiods)
            elem.lperiod=elem.l/elem.nperiods
            elem.Kx=eval(data_par[pos+2][2])
            elem.Ky=0
        elif elem.__class__ == Cavity:
            elem.l=eval(data_par[pos][2])
            elem.v=eval(data_par[pos+1][2])*1e-9  # V -> GV
            elem.phi=(eval(data_par[pos+2][2])-90)#/180.*np.pi
            elem.f=eval(data_par[pos+3][2])
            elem.delta_e=elem.v*cos(elem.phi)  # in GeV
        elif elem.__class__ == Hcor:
            elem.l=eval(data_par[pos][2])
        elif elem.__class__ == Vcor:
            elem.l=eval(data_par[pos][2])
        elif elem.__class__ == Monitor:
            elem.l=eval(data_par[pos][2])
        elif elem.__class__ == Marker:
            elem.l= 0. # eval(data_par[pos][2])
        elif elem.__class__ == Drift:
            elem.l=eval(data_par[pos][2])
        elem.id = elem.id.replace('.','_')
        elem.id = elem.id.replace('[','_')
        elem.id = elem.id.replace(']','')
    return lattice


"""
def insert_drifts(z_start, z_stop, lat_def):
    lattice= []
    s0=z_start
    for elem in lat_def:
        if elem.z>z_start and elem.z<z_stop:
            if s0==z_start:
                ds=elem.z-elem.l-s0
            else:
                ds=elem.s-elem.l-s0
            if ds>0:
                str_id=str(ds).replace('.','_')
                lattice=lattice+[Drift(l=ds,id='DRIFT_'+str_id)]
            lattice=lattice+[elem]
            print elem.id,elem.type,elem.z, elem.s, ds
            s0=elem.s
        elif elem.z>z_stop:
            break
    ds=z_stop-elem.z
    if ds>0:
        str_id=str(ds).replace('.','_')
        lattice=lattice+[Drift(l=ds,id='DRIFT'+str_id)]
    return lattice
"""

