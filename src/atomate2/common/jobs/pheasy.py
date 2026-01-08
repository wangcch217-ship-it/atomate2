"""Jobs for running phonon calculations with phonopy and pheasy."""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.io import read as ase_read
from emmet.core import __version__ as _emmet_core_version
from emmet.core.phonon import PhononBSDOSDoc
from hiphive import ClusterSpace, ForceConstantPotential, enforce_rotational_sum_rules
from hiphive import ForceConstants as HiPhiveForceConstants
from hiphive.cutoffs import estimate_maximum_cutoff
from hiphive.utilities import extract_parameters
from jobflow import job
from packaging.version import parse as parse_version
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_force_constants_to_hdf5
from phonopy.interface.vasp import write_vasp
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
from pymatgen.core import Structure
from pymatgen.io.phonopy import (
    get_ph_bs_symm_line,
    get_ph_dos,
    get_phonopy_structure,
    get_pmg_structure,
)
from pymatgen.io.vasp import Kpoints
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

from atomate2.common.jobs.phonons import _generate_phonon_object, _get_kpath

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D

logger = logging.getLogger(__name__)

############################################################################### 

import subprocess
import tempfile
import os
import shutil
import numpy as np
import logging

logger = logging.getLogger(__name__)

# 全局物理常数：Bohr到Angstrom的转换系数
BOHR_TO_ANGSTROM = 0.529177249


class ALM:
    """
    ALM (Anharmonic Lattice Model) 接口类
    
    使用 ALAMODE 2.0dev 版本，正确调用 ALM 可执行文件
    进行精确的力常数参数计算
    """
    
    def __init__(self, lattice, positions, numbers, alm_save_dir=None):
        """
        初始化 ALM 对象
        
        Parameters
        ----------
        lattice : array_like (3x3)
            晶格矢量矩阵，单位：Angstrom
        positions : array_like (N x 3)
            原子分数坐标
        numbers : array_like (N,)
            原子序数
        alm_save_dir : str, optional
            ALM日志保存目录（绝对路径），默认为当前工作目录
        """
        self.lattice = np.array(lattice)
        self.positions = np.array(positions)
        self.numbers = np.array(numbers)
        self.temp_dir = None
        self.original_dir = None
        self.alm_save_dir = alm_save_dir  # ALM日志保存目录
        
        # 使用系统 PATH 查找 alm
        self.alm_executable = shutil.which("alm")
        if self.alm_executable:
            logger.info(f"✅ 使用系统 ALM: {self.alm_executable}")
            self.use_fallback = False
        else:
            raise RuntimeError(
                "ALM 可执行文件未找到！\n"
                "请确保 'alm' 在系统 PATH 中。\n"
                "可以通过以下方式添加：\n"
                "  export PATH=/path/to/alamode/build/alm:$PATH"
            )
    
    def __enter__(self):
        """进入上下文管理器"""
        if not self.use_fallback:
            self.temp_dir = tempfile.mkdtemp(prefix="alm_work_")
            self.original_dir = os.getcwd()
            os.chdir(self.temp_dir)
            logger.info(f"📁 ALM 工作目录: {self.temp_dir}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器，保存ALM日志到指定目录"""
        if self.temp_dir and self.original_dir:
            # 使用用户指定的alm_save_dir，否则用当前目录
            save_dir = self.alm_save_dir if self.alm_save_dir else self.original_dir
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # 复制ALM文件
            # ALM 1.9.9+ 生成 alm.xml (通用名)，旧版本生成 alm_FCx.xml (特定阶数名)
            alm_files = [
                'alm.in', 'alm.log',
                'alm.xml', 'alm.h5', 'alm.fcs',  # ALM 1.9.9+ 格式
                'alm_FC2.xml', 'alm_FC3.xml', 'alm_FC4.xml'  # 旧版本格式（兼容）
            ]
            copied_files = []
            
            for fname in alm_files:
                src = os.path.join(self.temp_dir, fname)
                if os.path.exists(src):
                    dst = os.path.join(save_dir, fname)
                    shutil.copy2(src, dst)
                    copied_files.append(fname)
            
            os.chdir(self.original_dir)
            
            if copied_files:
                logger.info(f"✅ ALM 日志已保存到: {save_dir}")
                logger.info(f"   文件: {', '.join(copied_files)}")
            
            # 清理临时目录
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.debug(f"⚠️  临时目录保留: {self.temp_dir}")
    
    def define(self, max_order, cutoffs=None):
        """
        定义力常数的阶数和截断半径
        
        Parameters
        ----------
        max_order : int
            最大力常数阶数（1, 2, 3, 或 4）
        cutoffs : list of float, optional
            各阶力常数的截断半径，单位：Bohr
        """
        self.max_order = max_order
        self.cutoffs = cutoffs
        
        if self.use_fallback:
            logger.info(f"使用回退方法估算 {max_order} 阶力常数参数")
            return
        
        logger.info(f"📝 创建 ALM 输入文件 (max_order={max_order})")
        self._create_alm_input(max_order, cutoffs)
    
    def suggest(self):
        """运行 ALM 的 suggest 模式来确定独立参数数量"""
        if self.use_fallback:
            return
        
        logger.info("🚀 运行 ALM suggest 模式...")
        success = self._run_alm()
        
        if not success:
            logger.warning("⚠️  ALM 运行失败，切换到回退估算方法")
            self.use_fallback = True
    
    def _create_alm_input(self, max_order, cutoffs=None):
        """
        创建 ALM 输入文件（ALAMODE 2.0dev 格式）
        
        关键格式：
        1. NKD = X; KD = 元素符号  (用分号分隔)
        2. &cutoff 部分必需
        3. position: 种类索引 x y z
        """
        from ase.data import chemical_symbols
        
        natom = len(self.numbers)
        unique_numbers = sorted(set(self.numbers))
        nkd = len(unique_numbers)
        
        # 获取元素符号列表
        element_symbols = [chemical_symbols[num] for num in unique_numbers]
        
        with open("alm.in", "w") as f:
            # ===== &general 部分 =====
            f.write("&general\n")
            f.write(" PREFIX = alm\n")
            f.write(" MODE = suggest\n")
            f.write(" NAT = {}\n".format(natom))
            
            # 关键：使用分号分隔 NKD 和 KD
            f.write(" NKD = {}; KD = {}\n".format(nkd, " ".join(element_symbols)))
            
            f.write(" TOLERANCE = 1.0e-3\n")
            f.write(" PRINTSYM = 1\n")  # 输出对称性信息和独立参数数量
            f.write("/\n\n")
            
            # ===== &interaction 部分 =====
            f.write("&interaction\n")
            f.write(" NORDER = {}\n".format(max_order))
            f.write("/\n\n")
            
            # ===== &cutoff 部分（必需）=====
            # 关键修复：cutoff值必须在同一行，用空格分隔！
            f.write("&cutoff\n")

            if cutoffs and max_order >= 1:
                # 确定元素对格式
                if len(element_symbols) == 1:
                    # 单元素体系：Si-Si
                    pair_format = f"{element_symbols[0]}-{element_symbols[0]}"
                else:
                    # 多元素体系：使用通配符（适用于所有元素对）
                    pair_format = "*-*"

                # 构建一行cutoff：元素对 + 所有阶数的截断半径
                # NORDER=1 → 一行一个cutoff值 (2阶)
                # NORDER=2 → 一行两个cutoff值 (2阶, 3阶)
                # NORDER=3 → 一行三个cutoff值 (2阶, 3阶, 4阶)
                cutoff_line = f" {pair_format}"
                
                for order_index in range(max_order):
                    # order_index: 0,1,2,... 对应真实阶数 2,3,4,...
                    
                    if order_index < len(cutoffs):
                        cutoff_value = cutoffs[order_index]
                        
                        if cutoff_value is not None and cutoff_value > 0:
                            # 正值：转换为Angstrom
                            cutoff_angstrom = cutoff_value * BOHR_TO_ANGSTROM
                            cutoff_line += f" {cutoff_angstrom:.6f}"
                        else:
                            # None或非正值（如-1）：写 None
                            cutoff_line += " None"
                    else:
                        # 没有提供该阶的截断半径
                        cutoff_line += " None"
                
                # 写入一行
                f.write(cutoff_line + "\n")
            else:
                # max_order >= 1 但没有提供cutoffs
                if len(element_symbols) == 1:
                    pair_format = f"{element_symbols[0]}-{element_symbols[0]}"
                else:
                    pair_format = "*-*"

                # 写入一行，所有阶数都是 None
                cutoff_line = f" {pair_format}"
                for order_index in range(max(1, max_order)):
                    cutoff_line += " None"
                f.write(cutoff_line + "\n")

            f.write("/\n\n")
            # ===== &cell 部分 =====
            # 计算晶格常数（取第一个晶格矢量的模）
            lattice_const = np.linalg.norm(self.lattice[0])
            
            # 归一化晶格矢量
            normalized_lattice = self.lattice / lattice_const
            
            f.write("&cell\n")
            f.write("    {:.10f}\n".format(lattice_const))
            for vec in normalized_lattice:
                f.write("     {:20.15f}     {:20.15f}     {:20.15f}\n".format(
                    vec[0], vec[1], vec[2]))
            f.write("/\n\n")
            
            # ===== &position 部分 =====
            # 格式：种类索引 x y z (第一列是 1, 2, ..., 不是元素符号)
            # 确保分数坐标在 [0, 1) 范围内
            normalized_positions = self.positions % 1.0
            
            f.write("&position\n")
            for i, (num, pos) in enumerate(zip(self.numbers, normalized_positions), 1):
                kd = unique_numbers.index(num) + 1  # 种类索引从 1 开始
                f.write("   {}     {:20.15f}     {:20.15f}     {:20.15f}\n".format(
                    kd, pos[0], pos[1], pos[2]))
            f.write("/\n")
        
        logger.debug("✅ ALM 输入文件创建完成: alm.in")
    
    def _create_alm_fit_input(self, max_order, cutoffs=None):
        """
        创建ALM拟合输入文件（MODE=opt）
        
        与suggest模式的区别：
        - MODE = opt (而不是suggest)
        - 需要指定DFSET文件
        """
        from ase.data import chemical_symbols
        
        natom = len(self.numbers)
        unique_numbers = sorted(set(self.numbers))
        nkd = len(unique_numbers)
        element_symbols = [chemical_symbols[num] for num in unique_numbers]
        
        with open("alm.in", "w") as f:
            # &general
            f.write("&general\n")
            f.write(" PREFIX = alm\n")
            f.write(" MODE = opt\n")  # ✅ 拟合模式
            f.write(f" NAT = {natom}\n")
            f.write(f" NKD = {nkd}; KD = {' '.join(element_symbols)}\n")
            f.write(" TOLERANCE = 1.0e-3\n")
            f.write(" PRINTSYM = 1\n")  # 输出对称性信息和独立参数数量
            f.write("/\n\n")
            
            # &interaction
            f.write("&interaction\n")
            f.write(f" NORDER = {max_order}\n")
            f.write("/\n\n")
            
            # &cutoff
            f.write("&cutoff\n")
            if cutoffs and max_order >= 1:
                if len(element_symbols) == 1:
                    pair_format = f"{element_symbols[0]}-{element_symbols[0]}"
                else:
                    pair_format = "*-*"
                
                cutoff_line = f" {pair_format}"
                for order_index in range(max_order):
                    if order_index < len(cutoffs):
                        cutoff_value = cutoffs[order_index]
                        if cutoff_value is not None and cutoff_value > 0:
                            cutoff_angstrom = cutoff_value * BOHR_TO_ANGSTROM
                            cutoff_line += f" {cutoff_angstrom:.6f}"
                        else:
                            cutoff_line += " None"
                    else:
                        cutoff_line += " None"
                f.write(cutoff_line + "\n")
            else:
                if len(element_symbols) == 1:
                    pair_format = f"{element_symbols[0]}-{element_symbols[0]}"
                else:
                    pair_format = "*-*"
                cutoff_line = f" {pair_format}"
                for _ in range(max(1, max_order)):
                    cutoff_line += " None"
                f.write(cutoff_line + "\n")
            f.write("/\n\n")
            
            # &cell
            lattice_const = np.linalg.norm(self.lattice[0])
            normalized_lattice = self.lattice / lattice_const
            
            f.write("&cell\n")
            f.write(f" {lattice_const:.10f}\n")
            for vec in normalized_lattice:
                f.write(f" {vec[0]:20.15f} {vec[1]:20.15f} {vec[2]:20.15f}\n")
            f.write("/\n\n")
            
            # &position
            normalized_positions = self.positions % 1.0
            f.write("&position\n")
            for i, (num, pos) in enumerate(zip(self.numbers, normalized_positions), 1):
                kd = unique_numbers.index(num) + 1
                f.write(f" {kd} {pos[0]:20.15f} {pos[1]:20.15f} {pos[2]:20.15f}\n")
            f.write("/\n\n")
            
            # &optimize (拟合参数)
            f.write("&optimize\n")
            f.write(" DFSET = DFSET\n")  # ✅ 指定训练数据文件
            f.write("/\n")


    def _run_alm(self):
        """
        运行 ALM 可执行文件 - 修复版 + 增强调试

        Returns
        -------
        bool
            True 如果运行成功，False 否则
        """
        try:
            import os
            import subprocess
            import shutil
            import time

            # ===== 保存调试文件到指定目录 =====
            debug_dir = self.alm_save_dir if self.alm_save_dir else self.original_dir
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            if os.path.exists("alm.in"):
                try:
                    if not os.path.exists(debug_dir):
                        os.makedirs(debug_dir, exist_ok=True)
                    debug_file = os.path.join(debug_dir, f"alm_in_{timestamp}.txt")
                    shutil.copy("alm.in", debug_file)
                    logger.info(f"🔍 ALM 输入已保存: {debug_file}")
                except Exception as e:
                    logger.warning(f"⚠️  无法保存调试文件: {e}")
            # =====================================

            #定义 debug_enabled 变量
            debug_enabled = self.alm_save_dir is not None
            
            #创建环境变量副本
            env = os.environ.copy()
            
            # ===== 彻底移除导致 cmdModule 警告的所有环境变量 =====
            module_vars_to_remove = [
                # 基本module变量
                'LOADEDMODULES', '_LMFILES_', 'MODULE_VERSION', 
                'MODULE_VERSION_STACK', 'MODULEPATH', 'MODULESHOME',
                'MODULEPATH_ROOT', 'MODULEPATH_modshare', 
                'LOADEDMODULES_modshare', '_LMFILES__modshare',
                # Lmod相关变量
                'LMOD_CMD', 'LMOD_DIR', 'LMOD_PKG', 'LMOD_VERSION',
                'LMOD_sys', 'LMOD_arch', 'LMOD_SYSTEM_DEFAULT_MODULES',
                'LMOD_SETTARG_CMD', 'LMOD_FULL_SETTARG_SUPPORT',
                'LMOD_SETTARG_FULL_SUPPORT', 'LMOD_COLORIZE',
                'LMOD_PREPEND_BLOCK',
                # BASH函数
                'BASH_FUNC_module%%', 'BASH_FUNC_ml%%',
                'BASH_FUNC__module_raw%%', 'BASH_FUNC_switchml%%',
            ]

            for var in module_vars_to_remove:
                env.pop(var, None)
            
            # 删除所有以特定前缀开头的环境变量
            keys_to_remove = [
                k for k in list(env.keys())
                if (k.startswith('BASH_FUNC_') or 
                    k.startswith('LMOD') or
                    ('MODULE' in k.upper() and k not in ['MODULESHOME', 'MODULEPATH']))
            ]
            for key in keys_to_remove:
                env.pop(key, None)
            
            logger.debug(f"清理了 {len(module_vars_to_remove) + len(keys_to_remove)} 个Module相关变量")

            logger.info(f"🚀 运行 ALM: {self.alm_executable}")
            logger.debug(f"工作目录: {os.getcwd()}")

            # 运行 ALM（增加超时到 600 秒，适应大超胞）
            result = subprocess.run(
                [self.alm_executable, "alm.in"],
                capture_output=True,
                text=True,
                timeout=60000,  # 从 120 改为 600 秒
                env=env,
                cwd=os.getcwd()
            )

            # 保存完整日志
            log_file = "alm.log"
            with open(log_file, "w") as f:
                f.write("=== ALM 执行信息 ===\n")
                f.write(f"可执行文件: {self.alm_executable}\n")
                f.write(f"工作目录: {os.getcwd()}\n")
                f.write(f"退出码: {result.returncode}\n")
                f.write(f"\n=== STDOUT ===\n")
                f.write(result.stdout)
                f.write(f"\n=== STDERR ===\n")
                f.write(result.stderr)
            
            # 也保存到调试目录
            if debug_enabled:
                try:
                    shutil.copy(log_file, f"{debug_dir}/alm_log_{timestamp}.txt")
                    logger.info(f"🔍 ALM 日志已保存: {debug_dir}/alm_log_{timestamp}.txt")
                except Exception as e:
                    logger.warning(f"无法保存调试日志: {e}")

            success_markers = [
                "Job finished",
                "ALAMODE finished", 
                "Finished!",
                "Calculation finished"
            ]

            stdout_has_success = any(marker in result.stdout for marker in success_markers)

            # 2. 检查是否有参数输出
            has_fc_info = (
                "Number of free HARMONIC FCs" in result.stdout or
                "Number of free" in result.stdout and "FCs" in result.stdout or
                "nparams" in result.stdout
            )

            # 3. 检查输出文件
            output_files_exist = any(os.path.exists(f) for f in [
                "alm.pattern_HARMONIC",
                "alm_HARMONIC",
                "alm.fcs", 
                "alm_FC2.xml"
            ])

            # ✅ 判断成功的条件：任一即可
            if stdout_has_success or has_fc_info or output_files_exist:
                logger.info("✅ ALM 运行成功")

                # 如果退出码非零，记录警告但不影响结果
                if result.returncode != 0:
                    logger.warning(f"⚠️  ALM 退出码非零 ({result.returncode})，但输出正常")

                    # 分析 stderr 中的错误
                    stderr_lines = result.stderr.strip().split('\n') if result.stderr else []
                    real_errors = [line for line in stderr_lines 
                                  if line and 'cmdModule' not in line and 'ERROR' in line.upper()]

                    if real_errors:
                        logger.warning("发现以下错误信息:")
                        for line in real_errors[:3]:
                            logger.warning(f"  {line}")
                    else:
                        logger.info("只有 cmdModule 警告，可以忽略")

                return True

            # 如果没有成功标志，才认为失败
            logger.error("❌ ALM 运行失败")
            logger.error(f"退出码: {result.returncode}")

            # 打印关键错误
            if result.stderr:
                stderr_lines = result.stderr.strip().split('\n')
                error_lines = [line for line in stderr_lines 
                              if line and 'cmdModule' not in line]

                if error_lines:
                    logger.error("错误信息:")
                    for line in error_lines[:5]:
                        logger.error(f"  {line}")

            # 打印部分 stdout 帮助调试
            if result.stdout:
                logger.debug(f"STDOUT 前500字符:\n{result.stdout[:500]}")

            return False

        except subprocess.TimeoutExpired:
            logger.error("❌ ALM 运行超时（>600秒）")
            logger.error("   可能原因：超胞太大或计算复杂度过高")
            logger.error(f"   当前原子数: {len(self.numbers)}")
            return False
        except FileNotFoundError as e:
            logger.error(f"❌ 找不到文件: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ 运行 ALM 时出错: {type(e).__name__}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False


    def _get_number_of_irred_fc_elements(self, order):
        """
        获取指定阶数的独立力常数参数数量
        
        Parameters
        ----------
        order : int
            力常数阶数（1, 2, 3, 或 4）
        
        Returns
        -------
        int
            独立参数数量
        """
        if self.use_fallback:
            result = self._get_irred_fc_fallback(order)
            logger.info(f"📊 回退方法估算 {order} 阶参数数量: {result}")
            return result
        else:
            result = self._get_irred_fc_alm(order)
            logger.info(f"📊 ALM 计算 {order} 阶参数数量: {result}")
            return result
    
    def _get_irred_fc_alm(self, order):
        """从 ALM 输出中解析独立力常数参数数量
        
        注意：order参数的含义（与ALM输出关键字对应）：
        - order=1: HARMONIC（物理2阶力常数）
        - order=3: ANHARM3（物理3阶力常数）
        - order=4: ANHARM4（物理4阶力常数）
        没有order=2，因为ALM输出中没有ANHARM2关键字
        """
        try:
            if not os.path.exists("alm.log"):
                logger.warning("⚠️  ALM 日志文件不存在")
                return self._get_irred_fc_fallback(order)

            with open("alm.log", "r") as f:
                content = f.read()

            import re

            # 根据order值匹配对应的ALM输出关键字
            if order == 1:
                # HARMONIC（物理2阶力常数）
                patterns = [
                    rf"Number of free\s+HARMONIC FCs\s*:\s*(\d+)",
                    rf"Number of\s+HARMONIC FCs\s*:\s*(\d+)",
                ]
            elif order == 3:  # 注意：是3不是2！
                # ANHARM3（物理3阶力常数）
                patterns = [
                    rf"Number of free\s+ANHARM3 FCs\s*:\s*(\d+)",
                    rf"Number of\s+ANHARM3 FCs\s*:\s*(\d+)",
                ]
            elif order == 4:
                # ANHARM4（物理4阶力常数）
                patterns = [
                    rf"Number of free\s+ANHARM4 FCs\s*:\s*(\d+)",
                    rf"Number of\s+ANHARM4 FCs\s*:\s*(\d+)",
                ]
            else:
                logger.warning(f"不支持的order值: {order}，只支持1,3,4")
                return self._get_irred_fc_fallback(order)

            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    n_params = int(matches[0])
                    logger.info(f"✅ 从 ALM 输出解析 order={order}: {n_params}")
                    return n_params

            # 如果解析失败
            logger.warning(f"⚠️  无法从 ALM 输出中解析 order={order} 的参数数量")
            return self._get_irred_fc_fallback(order)

        except Exception as e:
            logger.warning(f"⚠️  解析 ALM 输出时出错: {e}")
            return self._get_irred_fc_fallback(order)
    
    def _get_irred_fc_fallback(self, order):
        """回退方法：使用数学估算计算独立参数数量
        
        注意：order参数的含义（与_get_irred_fc_alm保持一致）：
        - order=1: HARMONIC（物理2阶力常数）
        - order=3: ANHARM3（物理3阶力常数）
        - order=4: ANHARM4（物理4阶力常数）
        
        Parameters
        ----------
        order : int
            力常数编号（1, 3, 4）
        
        Returns
        -------
        int
            估算的独立参数数量
        """
        natoms = len(self.numbers)
        symmetry_factor = 0.2  # 典型晶体的对称性约化因子
        
        if order == 1:
            # HARMONIC（物理2阶力常数）
            base = 9 * natoms * natoms
            return max(1, int(base * symmetry_factor * 0.15))
        
        elif order == 3:
            # ANHARM3（物理3阶力常数）
            base = 27 * natoms ** 3
            return max(1, int(base * symmetry_factor * 0.008))
        
        elif order == 4:
            # ANHARM4（物理4阶力常数）
            base = 81 * natoms ** 4
            return max(1, int(base * symmetry_factor * 0.0005))
        
        else:
            logger.warning(f"Fallback不支持order={order}，只支持1,3,4")
            # 返回一个保守估计
            return max(1, int(natoms * 10))
###############################################################################

_DEFAULT_FILE_PATHS = {
    "force_displacements": "dataset_forces.npy",
    "displacements": "dataset_disps.npy",
    "displacements_folded": "dataset_disps_array_rr.npy",
    "phonopy": "phonopy.yaml",
    "band_structure": "phonon_band_structure.yaml",
    "band_structure_plot": "phonon_band_structure.pdf",
    "dos": "phonon_dos.yaml",
    "dos_plot": "phonon_dos.pdf",
    "force_constants": "FORCE_CONSTANTS",
    "harmonic_displacements": "disp_matrix.npy",
    "anharmonic_displacements": "disp_matrix_anhar.npy",
    "harmonic_force_matrix": "force_matrix.npy",
    "anharmonic_force_matrix": "force_matrix_anhar.npy",
    "website": "phonon_website.json",
}

#修改
def sanitize_complex(obj):
    """
    Recursively sanitize complex numbers by converting them to real parts (ignoring imaginary parts).
    """
    import numpy as np

    if isinstance(obj, complex):
        # Convert complex to real part only
        return float(obj.real)
    elif isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            # If NumPy array contains complex numbers, keep only real part
            return obj.real.tolist()
        return obj.tolist() if hasattr(obj, 'tolist') else obj
    elif isinstance(obj, (dict, list, tuple)):
        # Process dictionaries, lists, and tuples recursively
        if isinstance(obj, dict):
            return {k: sanitize_complex(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_complex(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(sanitize_complex(item) for item in obj)
    return obj
#修改

@job
def get_supercell_size(
    structure: Structure,
    min_length: float,
    max_atoms: int,
    force_90_degrees: bool,
    force_diagonal: bool,
) -> list[list[float]]:
    """
    Determine supercell size with given min_length and max_length.

    Parameters
    ----------
    structure: Structure Object
        Input structure that will be used to determine supercell
    min_length: float
        minimum length of cell in Angstrom
    max_length: float
        maximum length of cell in Angstrom
    prefer_90_degrees: bool
        if True, the algorithm will try to find a cell with 90 degree angles first
    allow_orthorhombic: bool
        if True, orthorhombic supercells are allowed
    **kwargs:
        Additional parameters that can be set.
    """
    transformation = CubicSupercellTransformation(
        min_length=min_length,
        max_atoms=max_atoms,
        force_90_degrees=force_90_degrees,
        force_diagonal=force_diagonal,
        angle_tolerance=1e-2,
        allow_orthorhombic=False,
    )
    transformation.apply_transformation(structure=structure)
    return transformation.transformation_matrix.transpose().tolist()


@job(data=[Structure])
def generate_phonon_displacements(
    structure: Structure,
    supercell_matrix: np.array,
    displacement: float,
    num_displaced_supercells: int,
    cal_3rd_order: bool,
    cal_4th_order: bool,
    cal_ther_cond: bool,
    displacement_anhar: float,
    num_disp_anhar: int,
    fcs_cutoff_radius: list[int],
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
    alm_save_dir: str | None = None,  
    random_seed: int | None = 103,
    verbose: bool = False,
) -> dict:
    """Generate displacements for harmonic and anharmonic phonon calculations."""
    
    # 使用传入的 alm_save_dir (不再从环境变量读取)
    if verbose and alm_save_dir:
        logger.info(f"ALM日志将保存到: {alm_save_dir}")
    
    # ========== 1. 生成谐波位移 ==========
    def add_magmom_to_supercell(supercell_struct, primitive_struct, sc_matrix):
        """
        将原胞的磁矩信息继承到超胞
        
        优先使用 pymatgen 超胞方法（更安全），失败时回退到简单重复
        
        Parameters
        ----------
        supercell_struct : Structure
            phonopy 生成的超胞结构
        primitive_struct : Structure
            原胞结构（包含 magmom site property）
        sc_matrix : np.ndarray
            超胞矩阵 (3x3)
            
        Returns
        -------
        Structure
            添加了磁矩信息的超胞结构
        """
        if not primitive_struct.site_properties.get("magmom"):
            logger.debug("原胞结构没有 magmom 属性，跳过磁矩继承")
            return supercell_struct
        
        try:
            # ===== 方法 1: 使用 pymatgen 超胞（推荐）=====
            logger.debug("尝试使用 pymatgen 超胞方法继承磁矩...")
            
            # 用 pymatgen 生成参考超胞（自动继承 magmom）
            reference_supercell = primitive_struct * sc_matrix
            
            # 验证原子数
            if len(reference_supercell) != len(supercell_struct):
                raise ValueError(
                    f"原子数不匹配: pymatgen={len(reference_supercell)}, "
                    f"phonopy={len(supercell_struct)}"
                )
            
            # 检查原子顺序是否一致
            order_match = True
            for site1, site2 in zip(reference_supercell, supercell_struct):
                # 检查元素
                if site1.specie != site2.specie:
                    order_match = False
                    break
                
                # 检查坐标（容差 0.1 Angstrom）
                dist = np.linalg.norm(site1.coords - site2.coords)
                if dist > 0.1:
                    order_match = False
                    break
            
            if order_match:
                # 顺序一致，直接使用 pymatgen 的 magmom
                ref_magmoms = reference_supercell.site_properties["magmom"]
                supercell_struct.add_site_property("magmom", ref_magmoms)
                
                logger.info(
                    f"✓ 磁矩继承（pymatgen自动）: "
                    f"{len(primitive_struct)} → {len(supercell_struct)} atoms"
                )
            else:
                # 顺序不一致，建立坐标映射
                logger.warning(
                    "原子顺序不完全匹配，建立坐标映射关系..."
                )
                
                ref_magmoms = reference_supercell.site_properties["magmom"]
                target_magmoms = []
                
                for target_site in supercell_struct:
                    target_coords = target_site.coords
                    target_species = target_site.specie
                    
                    # 在参考结构中找到匹配的原子
                    matched = False
                    for i, ref_site in enumerate(reference_supercell):
                        if ref_site.specie != target_species:
                            continue
                        
                        dist = np.linalg.norm(target_coords - ref_site.coords)
                        if dist < 0.1:
                            target_magmoms.append(ref_magmoms[i])
                            matched = True
                            break
                    
                    if not matched:
                        # 未找到匹配，使用默认值
                        logger.warning(
                            f"未找到匹配原子: {target_species} at "
                            f"{target_coords}, 使用磁矩 0.0"
                        )
                        target_magmoms.append(0.0)
                
                supercell_struct.add_site_property("magmom", target_magmoms)
                
                logger.info(
                    f"✓ 磁矩继承（坐标映射）: "
                    f"{len(primitive_struct)} → {len(supercell_struct)} atoms"
                )
            
            return supercell_struct
        
        except Exception as e:
            # 如果方法 1 失败，使用简单重复方法
            logger.warning(
                f"pymatgen 超胞方法失败: {e}. 使用简单重复方法..."
            )
            
            # 简单重复原胞的磁矩
            prim_magmoms = primitive_struct.site_properties["magmom"]
            n_repeat = len(supercell_struct) // len(primitive_struct)
            supercell_magmoms = prim_magmoms * n_repeat
            supercell_struct.add_site_property("magmom", supercell_magmoms)
            
            logger.info(
                f"✓ 磁矩继承（简单重复）: "
                f"{len(primitive_struct)} → {len(supercell_struct)} atoms"
            )
            
            return supercell_struct        
        
    phonon = _generate_phonon_object(
        structure,
        supercell_matrix,
        displacement,
        sym_reduce,
        symprec,
        use_symmetrized_structure,
        kpath_scheme,
        code,
        verbose=verbose,
    )

    supercell_ph = phonon.supercell
    lattice = supercell_ph.cell
    positions = supercell_ph.scaled_positions
    numbers = supercell_ph.numbers
    natom = len(numbers)

    # 使用 ALM 确定需要的位移数量
    with ALM(lattice, positions, numbers, alm_save_dir=alm_save_dir) as alm:
        alm.define(1)
        alm.suggest()
        n_fp = alm._get_number_of_irred_fc_elements(1)

    # 第一次保底：确保基础位移数至少为2(根据体系大小选择，100原子(3, 1.5)，100原子以上（3，1.2）
    num_disp_sc = max(3, int(np.ceil(n_fp / (3.0 * natom))))

    if verbose:
        logger.info(f"=== ALM 估算结果 ===")
        logger.info(f"原子数: {natom}")
        logger.info(f"自由参数数量: {n_fp}")
        logger.info(f"建议位移数量: {num_disp_sc}")
        logger.info(f"这将生成 {3 * natom * num_disp_sc} 个方程")
        logger.info(
            f"There are {n_fp} free parameters for the second-order force constants (FCs). "
            f"There are {3 * natom * num_disp_sc} equations to obtain the second-order FCs."
        )

    # 生成谐波位移,总是生成大于等于5的随机位移，无论对称性如何
    phonon.generate_displacements(
        distance=displacement,
        number_of_snapshots=(
            num_displaced_supercells
            if num_displaced_supercells != 0
            else max(3, int(np.ceil(num_disp_sc * 1.8)) + 1)  #  第二次保底：应用1.8倍过定系数后再次确保至少3个
        ),
        random_seed=random_seed,
    )
    
    # 转换为 pymatgen Structure 并继承磁矩
    supercells = phonon.supercells_with_displacements
    displacements = []
    
    for cell in supercells:
        disp_struct = get_pmg_structure(cell)
        # 使用改进的磁矩继承方法，传入 supercell_matrix
        disp_struct = add_magmom_to_supercell(
            disp_struct, 
            structure,
            supercell_matrix  # 关键参数
        )
        displacements.append(disp_struct)

    n_harmonic = len(displacements)
    logger.info(
        f"生成了 {n_harmonic} 个谐波位移超胞"
    )
    
    # ========== 2. 生成非谐位移 ==========
    n_anharmonic = 0
    if cal_3rd_order or cal_4th_order or cal_ther_cond:
        logger.info("=" * 80)
        if cal_4th_order:
            logger.info("生成非谐位移（用于四阶力常数：2+3+4 阶）")
        elif cal_3rd_order:
            logger.info("生成非谐位移（用于三阶力常数：2+3 阶）")
        else:
            logger.info("生成非谐位移（用于热导率：2+3 阶）")
        logger.info("=" * 80)
        
        # 根据计算需求确定最大阶数
        # ALAMODE的NORDER定义:
        #   NORDER=1: 计算2阶力常数
        #   NORDER=2: 计算2阶+3阶力常数
        #   NORDER=3: 计算2阶+3阶+4阶力常数
        if cal_4th_order:
            alm_max_order = 3  # 需要4阶 → NORDER=3
        elif cal_3rd_order or cal_ther_cond:
            alm_max_order = 2  # 只需要3阶 → NORDER=2
        else:
            alm_max_order = 1  # 只需要2阶 → NORDER=1

        with ALM(lattice, positions, numbers, alm_save_dir=alm_save_dir) as alm:
            alm.define(alm_max_order, fcs_cutoff_radius)
            alm.suggest()

            if cal_4th_order:
                # 分别获取3阶和4阶参数数量
                n_3rd = alm._get_number_of_irred_fc_elements(3)
                n_4th = alm._get_number_of_irred_fc_elements(4)
                
                # 分别计算基础位移数
                num_d_3rd_base = max(3, int(np.ceil(n_3rd / (3.0 * natom))))
                num_d_4th_base = max(3, int(np.ceil(n_4th / (3.0 * natom))))

                # 应用过定系数（修正为1.2倍，与三阶逻辑一致）
                num_d_3rd = max(3, int(np.ceil(num_d_3rd_base * 1.6)) + 1)
                num_d_4th = max(3, int(np.ceil(num_d_4th_base * 1.6)) + 1)
                
                # 相加得到总的非谐位移超胞数
                num_d_anh = num_d_3rd + num_d_4th
                
                logger.info(f"   3阶力常数参数: {n_3rd}")
                logger.info(f"   3阶基础位移: {num_d_3rd_base}, 过定位移: {num_d_3rd}")
                logger.info(f"   4阶力常数参数: {n_4th}")
                logger.info(f"   4阶基础位移: {num_d_4th_base}, 过定位移: {num_d_4th}")
                logger.info(f"   总非谐位移超胞数: {num_d_anh}")
            else:
                # 只计算3阶参数（2阶已固定）
                n_rd_anh = alm._get_number_of_irred_fc_elements(3)
                num_d_anh_base = max(3, int(np.ceil(n_rd_anh / (3.0 * natom))))
                # 应用过定系数（和2阶逻辑一致：1.6倍 + 1）
                num_d_anh = max(3, int(np.ceil(num_d_anh_base * 1.6)) + 1)
                
                logger.info(f"   3阶力常数参数: {n_rd_anh}")
                logger.info(f"   3阶基础位移: {num_d_anh_base}, 过定位移: {num_d_anh}")

        # 注意：num_d_anh 已经包含过定系数（1.2倍 + 1），无需再次应用
        num_dis_cells_anhar = (
            num_disp_anhar
            if num_disp_anhar != 0
            else num_d_anh  # ✅ 直接使用已包含过定系数的值
        )
        
        logger.info(f"ALM建议非谐位移数（含过定系数）: {num_d_anh}")
        logger.info(f"实际使用非谐位移数: {num_dis_cells_anhar}")

        # ✅ 关键：重新创建 phonon 对象
        phonon_anhar = _generate_phonon_object(
            structure,
            supercell_matrix,
            displacement_anhar,
            sym_reduce,
            symprec,
            use_symmetrized_structure,
            kpath_scheme,
            code,
            verbose=verbose,
        )
        
        phonon_anhar.generate_displacements(
            distance=displacement_anhar,
            number_of_snapshots=num_dis_cells_anhar,
            random_seed=random_seed,
        )
        
        supercells_anhar = phonon_anhar.supercells_with_displacements
        anharmonic_disps = []

        # ✅ 使用循环显式调用 add_magmom_to_supercell
        for cell in supercells_anhar:
            disp_struct = get_pmg_structure(cell)
            # 继承磁矩（与谐波部分逻辑完全一致）
            disp_struct = add_magmom_to_supercell(
                disp_struct,
                structure,
                supercell_matrix  # 必须传入超胞矩阵
            )
            anharmonic_disps.append(disp_struct)

        n_anharmonic = len(anharmonic_disps)

        logger.info(
            f"生成了 {n_anharmonic} 个非谐波位移超胞"
        )

        # ✅ 扩展列表而不是覆盖
        displacements.extend(anharmonic_disps)

    # ========== 3. 添加平衡结构 ==========
    equilibrium_struct = get_pmg_structure(phonon.supercell)
    equilibrium_struct = add_magmom_to_supercell(
        equilibrium_struct,
        structure,
        supercell_matrix
    )
    displacements.append(equilibrium_struct)
    
    metadata = {
        'n_harmonic': n_harmonic,
        'n_anharmonic': n_anharmonic,
    }
    
    return {
        "structures": displacements,
        "metadata": {
            "n_harmonic": n_harmonic,
            "n_anharmonic": n_anharmonic,
        }
    }
    
    
@job(
    output_schema=PhononBSDOSDoc,
    data=[PhononDos, PhononBandStructureSymmLine, "force_constants"],
)
def generate_frequencies_eigenvectors(
    structure: Structure,
    supercell_matrix: np.ndarray,
    displacement: float,
    # ==================== 必需参数——这些参数调用时必须提供，没有默认值 ====================
    cal_3rd_order: bool,
    cal_4th_order: bool,
    cal_ther_cond: bool,
    renorm_phonon: bool,
    renorm_temp: list[int],
    kpath_scheme: str,
    code: str,
    displacement_data: dict[str, list],
    displacement_metadata: dict,
    total_dft_energy: float,
    fcs_cutoff_radius: list[float],
    # ==================== 可选参数 ====================
    # 这些参数有默认值，调用时可以不提供
    anphon_ther_cond: bool = False,
    displacement_anhar: float = 0.03,
    renorm_mesh: list[int] | None = None,
    ther_cond_mesh: list[int] | None = None,
    ther_cond_temp: list[int] | None = None,
    sym_reduce: bool = True,
    symprec: float = 1e-5,
    use_symmetrized_structure: str | None = None,
    epsilon_static: Matrix3D | None = None,  # 统一用 | None 格式
    born: Matrix3D | None = None,            # 统一用 | None 格式
    **kwargs,
) -> PhononBSDOSDoc:
    """
    Analyze the phonon runs and summarize the results.

    Parameters
    ----------
    structure: Structure object
        Fully optimized structure used for phonon runs
    supercell_matrix: np.array
        array to describe supercell
    displacement: float
        displacement in Angstrom used for supercell computation
    sym_reduce: bool
        if True, symmetry will be used in phonopy
    symprec: float
        precision to determine symmetry
    use_symmetrized_structure: str
        primitive, conventional, None are allowed
    kpath_scheme: str
        kpath scheme for phonon band structure computation
    code: str
        code to run computations
    displacement_data: dict
        outputs from displacements
    total_dft_energy: float
        total DFT energy in eV per cell
    epsilon_static: Matrix3D
        The high-frequency dielectric constant
    born: Matrix3D
        Born charges
    verbose : bool = False
        Whether to log error messages.
    kwargs: dict
        Additional parameters that are passed to PhononBSDOSDoc.from_forces_born
    """
    # ========== 参数验证==========
    logger.info("=" * 80)
    logger.info("验证输入参数")
    logger.info("=" * 80)
    
    # 验证1: 声子重整化需要四阶力常数
    if renorm_phonon and not cal_4th_order:
        logger.error("参数冲突: renorm_phonon=True 需要cal_4th_order=True")
        raise ValueError(
            "声子重整化需要四阶力常数（2+3+4 阶）！\n"
             "请设置: cal_4th_order=True"
        )
    
    # 验证2: 截断半径检查
    if cal_3rd_order or cal_4th_order or cal_ther_cond:
        if len(fcs_cutoff_radius) < 2:
            raise ValueError(
                f"非谐计算需要至少 [2阶, 3阶] 截断半径\n"
                f"当前: {fcs_cutoff_radius}"
            )
        
        if cal_4th_order and len(fcs_cutoff_radius) < 3:
            logger.warning("四阶截断半径未定义，使用默认值 10 Bohr")
            fcs_cutoff_radius.append(10)
    
    # 显示配置
    if cal_4th_order:
        logger.info(f"   ├─ 四阶力常数: 是 (2+3+4 阶)")
    elif cal_3rd_order:
        logger.info(f"   ├─ 三阶力常数: 是 (2+3 阶)")
    else:
        logger.info(f"   ├─ 非谐效应: 否")
    
    logger.info(f"   ├─ 热导率: {'是' if cal_ther_cond else '否'}")
    logger.info(f"   └─ 声子重整化: {'是' if renorm_phonon else '否'}")
    logger.info("=" * 80)
    logger.info("")
    
    phonon = _generate_phonon_object(
        structure,
        supercell_matrix,
        displacement,
        sym_reduce,
        symprec,
        use_symmetrized_structure,
        kpath_scheme,
        code,
        verbose=False,
    )

    # Write the POSCAR and SPOSCAR files for the input of pheasy code
    supercell = phonon._supercell  # noqa: SLF001
    write_vasp("POSCAR", get_phonopy_structure(structure))
    write_vasp("SPOSCAR", supercell)

    # get the force-displacement dataset from previous calculations
    dataset_forces = np.array(displacement_data["forces"])
    np.save(_DEFAULT_FILE_PATHS["force_displacements"], dataset_forces)

    # To deduct the residual forces on an equilibrium structure to eliminate the
    # fitting error
    dataset_forces_array_rr = dataset_forces - dataset_forces[-1, :, :]

    # force matrix on the displaced structures
    dataset_forces_array_disp = dataset_forces_array_rr[:-1, :, :]

    # To handle the large dispalced distance in the dataset
    dataset_disps = np.array(
        [disps.frac_coords for disps in displacement_data["displaced_structures"]]
    )
    np.save(_DEFAULT_FILE_PATHS["displacements"], dataset_disps)

    dataset_disps_array_rr = np.round(
        (dataset_disps - supercell.scaled_positions), decimals=16
    )
    np.save(_DEFAULT_FILE_PATHS["displacements_folded"], dataset_disps_array_rr)

    dataset_disps_array_rr = np.where(
        dataset_disps_array_rr > 0.5,
        dataset_disps_array_rr - 1.0,
        dataset_disps_array_rr,
    )
    dataset_disps_array_rr = np.where(
        dataset_disps_array_rr < -0.5,
        dataset_disps_array_rr + 1.0,
        dataset_disps_array_rr,
    )

    # Transpose the displacement array on the
    # last two axes (atoms and coordinates)
    dataset_disps_array_rr_transposed = np.transpose(dataset_disps_array_rr, (0, 2, 1))

    # Perform matrix multiplication with the transposed supercell.cell
    # 'ij' for supercell.cell.T and
    # 'nkj' for the transposed dataset_disps_array_rr
    dataset_disps_array_rr_cartesian = np.einsum(
        "ij,njk->nik", supercell.cell.T, dataset_disps_array_rr_transposed
    )
    # Transpose back to the original format
    dataset_disps_array_rr_cartesian = np.transpose(
        dataset_disps_array_rr_cartesian, (0, 2, 1)
    )

    dataset_disps_array_use = dataset_disps_array_rr_cartesian[:-1, :, :]
    
    # ============ 修改：使用传入的元数据 ============
    # 直接使用从 generate_phonon_displacements 传来的信息
    num_har = displacement_metadata['n_harmonic']


    np.save(
        _DEFAULT_FILE_PATHS["harmonic_displacements"],
        dataset_disps_array_use[:num_har, :, :],
    )
    np.save(
        _DEFAULT_FILE_PATHS["harmonic_force_matrix"],
        dataset_forces_array_disp[:num_har, :, :],
    )

    # get the born charges and dielectric constant
    if born is not None and epsilon_static is not None:
        if len(structure) == len(born):
            borns, epsilon = symmetrize_borns_and_epsilon(
                ucell=phonon.unitcell,
                borns=np.array(born),
                epsilon=np.array(epsilon_static),
                symprec=symprec,
                primitive_matrix=phonon.primitive_matrix,
                supercell_matrix=phonon.supercell_matrix,
                is_symmetry=kwargs.get("symmetrize_born", True),
            )
        else:
            raise ValueError(
                "Number of born charges does not agree with number of atoms"
            )

        if code == "vasp" and not np.all(np.isclose(borns, 0.0)):
            phonon.nac_params = {
                "born": borns,
                "dielectric": epsilon,
                "factor": 14.399652,
            }
        # Other codes could be added here

    else:
        borns = None
        epsilon = None

    prim = ase_read("POSCAR")
    supercell = ase_read("SPOSCAR")

    # ========== 处理2阶截断半径参数 ==========
    c2_cutoff_str = ""
    if fcs_cutoff_radius and fcs_cutoff_radius[0] is not None:
        if fcs_cutoff_radius[0] > 0:
            # 正值：Bohr → Angstrom 转换
            c2_cutoff_angstrom = fcs_cutoff_radius[0] * BOHR_TO_ANGSTROM
            c2_cutoff_str = f"--c2 {c2_cutoff_angstrom:.6f}"
            logger.info(f"✅ 2阶截断: {fcs_cutoff_radius[0]:.2f} Bohr = {c2_cutoff_angstrom:.3f} Å")
        elif fcs_cutoff_radius[0] == -1:
            # -1：最大截断（不传参数，使用 Pheasy 默认 None）
            c2_cutoff_str = ""
            logger.info("✅ 2阶截断: 最大截断（-1 → None）")
        else:
            # 其他负值：不支持
            raise ValueError(
                f"不支持的2阶截断值: {fcs_cutoff_radius[0]}\n"
                f"请使用: 正值（距离/Bohr）、-1（最大截断）或 None（最大截断）"
            )
    else:
        # None 或未指定：不传参数（使用 Pheasy 默认值 None）
        c2_cutoff_str = ""
        logger.info("ℹ️  2阶截断: 最大截断（None）")

    # Create the clusters and orbitals for second order force constants
    # For the variables: --w, --nbody, they are used to specify the order of the
    # force constants. in the near future, we will add the option to specify the
    # order of the force constants. And these two variables can be defined by the
    # users.
    pheasy_cmd_1 = (
        f"pheasy --dim {int(supercell_matrix[0][0])} "
        f"{int(supercell_matrix[1][1])} "
        f"{int(supercell_matrix[2][2])} "
        f"-s -w 2 --symprec {float(symprec)} --nbody 2 {c2_cutoff_str}"
    )

    # Create the null space to further reduce the free parameters for
    # specific force constants and make them physically correct.
    pheasy_cmd_2 = (
        f"pheasy --dim {int(supercell_matrix[0][0])} "
        f"{int(supercell_matrix[1][1])} "
        f"{int(supercell_matrix[2][2])} -c --symprec "
        f"{float(symprec)} -w 2 {c2_cutoff_str}"
    )

    # Generate the Compressive Sensing matrix,i.e., displacement matrix
    # for the input of machine leaning method.i.e., LASSO,
    pheasy_cmd_3 = (
        f"pheasy --dim {int(supercell_matrix[0][0])} "
        f"{int(supercell_matrix[1][1])} "
        f"{int(supercell_matrix[2][2])} -w 2 -d "
        f"--symprec {float(symprec)} "
        f"--ndata {int(num_har)} --disp_file {c2_cutoff_str}"
    )

    # Here we set a criteria to determine which method to use to generate the
    # force constants. If the number of displacements is larger than 3, we
    # will use the LASSO method to generate the force constants. Otherwise,
    # we will use the least-squred method to generate the force constants.
    if len(phonon.displacements) > 3:
        # Calculate the force constants using the LASSO method due to the
        # random-displacement method Obviously, the rotaional invariance
        # constraint, i.e., tag: --rasr BHH, is enforced during the
        # fitting process.
        pheasy_cmd_4 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} -f --full_ifc "
            f"-w 2 --symprec {float(symprec)} "
            f"-l LASSO --std --rasr BHH --ndata {int(num_har)} {c2_cutoff_str}"
        )

    else:
        # Calculate the force constants using the least-squred method
        pheasy_cmd_4 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} -f --full_ifc "
            f"-w 2 --symprec {float(symprec)} "
            f"--rasr BHH --ndata {int(num_har)} {c2_cutoff_str}"
        )

    logger.info("Start running pheasy in cluster")

# Fix NumPy 2.x compatibility issue
    import math
    if not hasattr(np, 'math'):
        np.math = math
        logger.info("Applied NumPy 2.x compatibility fix")

    # Create pickle files that PHEASY expects
    import pickle
    with open('disp_matrix.pkl', 'wb') as f:
        pickle.dump(dataset_disps_array_use[:num_har, :, :], f)

    with open('force_matrix.pkl', 'wb') as f:
        pickle.dump(dataset_forces_array_disp[:num_har, :, :], f)

    logger.info(f"Created pickle files for PHEASY with {num_har} configurations")

    subprocess.call(shlex.split(pheasy_cmd_1))
    subprocess.call(shlex.split(pheasy_cmd_2))
    subprocess.call(shlex.split(pheasy_cmd_3))
    subprocess.call(shlex.split(pheasy_cmd_4))

    # When this code is run on Github tests, it is failing because it is
    # not able to find the FORCE_CONSTANTS file. This is because the file is
    # somehow getting generated in some temp directory. Can you fix the bug?
    fc_file = _DEFAULT_FILE_PATHS["force_constants"]

    # 验证3: cal_4th_order 需要 cal_3rd_order
    if cal_4th_order and not cal_3rd_order:
        raise ValueError(
            "cal_4th_order=True 需要 cal_3rd_order=True\n"
            "四阶力常数计算依赖三阶力常数"
        )

    # ✅ 计算非谐性力常数（完整或热导率模式）
    if cal_3rd_order or cal_4th_order or cal_ther_cond:
        # ✅ 直接从 metadata 获取，而不是推导
        num_anhar = displacement_metadata.get('n_anharmonic', 0)

        # ✅ 验证数据一致性
        expected_total = num_har + num_anhar
        actual_total = dataset_forces_array_disp.shape[0]

        if expected_total != actual_total:
            if actual_total < num_har:
                raise ValueError(
                    f"位移数据严重不足！\n"
                    f"  - 预期至少 {num_har} 个谐波位移\n"
                    f"  - 实际只有 {actual_total} 个总位移\n"
                    f"数据已损坏，无法继续计算"
                )
            logger.warning(
                f"数据不一致警告:\n"
                f"  - 预期位移数: {expected_total} (谐波: {num_har}, 非谐: {num_anhar})\n"
                f"  - 实际位移数: {actual_total}\n"
                f"  - 差值: {actual_total - expected_total}\n"
                f"将自动调整非谐位移数为: {actual_total - num_har}"
            )
            # ✅ 使用实际值，但记录警告
            num_anhar = actual_total - num_har
        if num_anhar > 0:
            logger.info("=" * 80)

            # 根据模式设置计算参数
            if cal_4th_order:
                # 模式1: 四阶力常数（2+3+4阶）
                max_order = 4
                nbody_str = "2 3 4"
                
                # ========== 3阶截断处理（与2阶逻辑一致）==========
                # 处理3阶截断半径
                c3_cutoff_str = ""
                if fcs_cutoff_radius and len(fcs_cutoff_radius) > 1:
                    if fcs_cutoff_radius[1] is not None:
                        if fcs_cutoff_radius[1] > 0:
                            # 正值：Bohr → Angstrom
                            c3_cutoff_angstrom = fcs_cutoff_radius[1] * BOHR_TO_ANGSTROM
                            c3_cutoff_str = f"--c3 {c3_cutoff_angstrom:.6f}"
                            logger.info(f"   - 3阶截断: {fcs_cutoff_radius[1]:.2f} Bohr = {c3_cutoff_angstrom:.3f} Å")
                        elif fcs_cutoff_radius[1] == -1:
                            # -1：最大截断
                            c3_cutoff_str = ""
                            logger.info("   - 3阶截断: 最大截断（-1 → None）")
                        else:
                            raise ValueError(f"不支持的3阶截断值: {fcs_cutoff_radius[1]}")
                    else:
                        # None：最大截断
                        c3_cutoff_str = ""
                        logger.info("   - 3阶截断: 最大截断（None）")
                else:
                    c3_cutoff_str = ""
                    logger.info("   - 3阶截断: 最大截断（未指定）")
                
                # ========== 4阶截断处理（与2阶逻辑一致）==========
                # 处理4阶截断半径
                c4_cutoff_str = ""
                if fcs_cutoff_radius and len(fcs_cutoff_radius) > 2:
                    if fcs_cutoff_radius[2] is not None:
                        if fcs_cutoff_radius[2] > 0:
                            # 正值：Bohr → Angstrom
                            c4_cutoff_angstrom = fcs_cutoff_radius[2] * BOHR_TO_ANGSTROM
                            c4_cutoff_str = f"--c4 {c4_cutoff_angstrom:.6f}"
                            logger.info(f"   - 4阶截断: {fcs_cutoff_radius[2]:.2f} Bohr = {c4_cutoff_angstrom:.3f} Å")
                        elif fcs_cutoff_radius[2] == -1:
                            # -1：最大截断
                            c4_cutoff_str = ""
                            logger.info("   - 4阶截断: 最大截断（-1 → None）")
                        else:
                            raise ValueError(f"不支持的4阶截断值: {fcs_cutoff_radius[2]}")
                    else:
                        # None：最大截断
                        c4_cutoff_str = ""
                        logger.info("   - 4阶截断: 最大截断（None）")
                else:
                    c4_cutoff_str = ""
                    logger.info("   - 4阶截断: 最大截断（未指定）")
                
                cutoff_str = f"{c3_cutoff_str} {c4_cutoff_str}".strip()
                logger.info("计算四阶力常数（2+3+4 阶）")
                logger.info("   - 拟合策略: 固定2阶，同时拟合3阶+4阶")

            elif cal_3rd_order or cal_ther_cond:
                # 模式2: 三阶力常数（2+3阶）
                max_order = 3
                nbody_str = "2 3"
                
                # ========== 3阶截断处理（与2阶逻辑一致）==========
                # 处理3阶截断半径
                c3_cutoff_str = ""
                if fcs_cutoff_radius and len(fcs_cutoff_radius) > 1:
                    if fcs_cutoff_radius[1] is not None:
                        if fcs_cutoff_radius[1] > 0:
                            # 正值：Bohr → Angstrom
                            c3_cutoff_angstrom = fcs_cutoff_radius[1] * BOHR_TO_ANGSTROM
                            c3_cutoff_str = f"--c3 {c3_cutoff_angstrom:.6f}"
                            logger.info(f"   - 3阶截断: {fcs_cutoff_radius[1]:.2f} Bohr = {c3_cutoff_angstrom:.3f} Å")
                        elif fcs_cutoff_radius[1] == -1:
                            # -1：最大截断
                            c3_cutoff_str = ""
                            logger.info("   - 3阶截断: 最大截断（-1 → None）")
                        else:
                            raise ValueError(f"不支持的3阶截断值: {fcs_cutoff_radius[1]}")
                    else:
                        # None：最大截断
                        c3_cutoff_str = ""
                        logger.info("   - 3阶截断: 最大截断（None）")
                else:
                    c3_cutoff_str = ""
                    logger.info("   - 3阶截断: 最大截断（未指定）")

                    # ✅✅✅ 添加这两行！！！
                cutoff_str = c3_cutoff_str
                logger.info("计算三阶力常数（用于热导率：2+3 阶）")
                logger.info("   - 拟合策略: 固定2阶，拟合3阶")

            logger.info("=" * 80)
            logger.info(f"谐波位移数: {num_har}")
            logger.info(f"非谐位移数: {num_anhar}")
            logger.info(f"总位移数: {num_har + num_anhar}")

            # 保存非谐位移和力矩阵
            np.save(
                _DEFAULT_FILE_PATHS["anharmonic_displacements"],
                dataset_disps_array_use[num_har:, :, :],
            )
            np.save(
                _DEFAULT_FILE_PATHS["anharmonic_force_matrix"],
                dataset_forces_array_disp[num_har:, :, :],
            )

            # 构建pheasy命令
            pheasy_cmd_5 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -s -w {max_order} --symprec "
                f"{float(symprec)} "
                f"--nbody {nbody_str} {cutoff_str}"
            )

            pheasy_cmd_6 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -c --symprec "
                f"{float(symprec)} -w {max_order} {cutoff_str}"
            )

            pheasy_cmd_7 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -w {max_order} -d --symprec "
                f"{float(symprec)} "
                f"--ndata {int(num_anhar)} --disp_file "
                f"{cutoff_str}" 
            )

            pheasy_cmd_8 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -f -w {max_order} --fix_fc2 " #--fix_fc2 固定2阶力常数，只优化3阶/4阶
                f"--symprec {float(symprec)} "
                f"-l LASSO --std --rasr BHH "  # 对所有阶数使用LASSO和旋转不变性约束
                f"--ndata {int(num_anhar)} "
                f"{cutoff_str}"
            )

            # 执行pheasy命令
            subprocess.call(shlex.split(pheasy_cmd_5))
            subprocess.call(shlex.split(pheasy_cmd_6))
            subprocess.call(shlex.split(pheasy_cmd_7))
            subprocess.call(shlex.split(pheasy_cmd_8))

            logger.info(f"Anharmonic force constants (up to {max_order}-order) calculation completed")
            
        else:
            logger.warning("="*60)
            if cal_4th_order:
                logger.warning("cal_4th_order=True but no anharmonic displacement data found!")
            elif cal_3rd_order:
                logger.warning("cal_3rd_order=True but no anharmonic displacement data found!")
            elif cal_ther_cond:
                logger.warning("cal_ther_cond=True but no anharmonic displacement data found!")
            logger.warning(f"Total displacements: {dataset_forces_array_disp.shape[0]}")
            logger.warning(f"Harmonic displacements: {num_har}")
            logger.warning(f"Anharmonic displacements: {num_anhar}")
            logger.warning("Skipping anharmonic force constants calculation")
            logger.warning("="*60)
    
    else:
        # 模式3: 纯谐波（2阶）
        logger.info("=" * 80)
        logger.info("纯谐波模式：仅计算二阶力常数")
        logger.info("=" * 80)


    # 声子重整化（仅在四阶力常数模式下可选）
    if renorm_phonon:
        if not cal_4th_order:
            raise ValueError("需要cal_4th_order=True")
        
        # 决定使用哪个mesh
        if renorm_mesh is not None:
            # 用户显式指定了renorm_mesh
            mesh_to_use = renorm_mesh
            logger.info(f"使用renorm_mesh: {mesh_to_use}")
        elif ther_cond_mesh is not None:
            # 用户没指定renorm_mesh，回退到ther_cond_mesh
            mesh_to_use = ther_cond_mesh
            logger.info(f"renorm_mesh未指定，使用ther_cond_mesh: {mesh_to_use}")
        else:
            # 两个都没有
            raise ValueError(
                "声子重整化需要mesh参数！\n"
                "请设置 renorm_mesh=[20,20,20] 或 ther_cond_mesh=[20,20,20]"
            )
        
        mesh_str = f"--mesh {mesh_to_use[0]} {mesh_to_use[1]} {mesh_to_use[2]}"
        
        if not renorm_temp:
            renorm_temp = [300]
        
        # 备份原始二阶力常数
        if os.path.exists("FORCE_CONSTANTS"):
            if not os.path.exists("FORCE_CONSTANTS_0K"):
                shutil.copy("FORCE_CONSTANTS", "FORCE_CONSTANTS_0K")
                logger.info("✅ 已备份原始0K二阶力常数: FORCE_CONSTANTS_0K")
            else:
                logger.info("ℹ️  FORCE_CONSTANTS_0K 已存在，跳过备份")
        
        logger.info("=" * 80)
        logger.info("开始声子重整化（SCAILD）")
        logger.info(f"温度: {renorm_temp} K")
        logger.info(f"k点网格: {mesh_to_use}")
        logger.info("=" * 80)
        
        renorm_success_count = 0

        for temp in renorm_temp:
            logger.info(f"T = {temp} K ...")
            
            pheasy_cmd = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} "
                f"-m SCAILD "
                f"-f -w 4 "
                f"--fix_fc2 "
                f"--temp {temp} "
                f"{mesh_str} "
                f"--symprec {float(symprec)} "
                f"-l LASSO --std --rasr BHH "
                f"--ndata {int(num_anhar)} "
                f"--hdf5"
            )
            
            try:
                result = subprocess.call(shlex.split(pheasy_cmd))
                
                if result != 0:
                    logger.warning(f"⚠️  T={temp}K SCAILD退出码: {result}")
                
                # 检查输出文件
                if os.path.exists("FORCE_CONSTANTS"):
                    output_name = f"FORCE_CONSTANTS_renorm_{temp}K"
                    shutil.copy("FORCE_CONSTANTS", output_name)
                    logger.info(f"✅ T={temp}K 完成，已保存: {output_name}")
                    
                    # 同时保存HDF5格式（如果存在）
                    if os.path.exists("fc2.hdf5"):
                        shutil.copy("fc2.hdf5", f"fc2_renorm_{temp}K.hdf5")
                        logger.info(f"   HDF5格式: fc2_renorm_{temp}K.hdf5")
                    
                    renorm_success_count += 1
                else:
                    logger.error(f"❌ T={temp}K 失败：未生成FORCE_CONSTANTS文件")
                    
            except Exception as e:
                logger.error(f"❌ T={temp}K 执行失败: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        logger.info("=" * 80)
        logger.info(f"声子重整化计算完成: {renorm_success_count}/{len(renorm_temp)} 个温度成功")
        logger.info("=" * 80)

    # begin to convert the force constants to the phonopy and phono3py format
    # for the further lattice thermal conductivity calculations
    if cal_ther_cond:
        # 检查必需参数
        if ther_cond_mesh is None or len(ther_cond_mesh) < 3:
            raise ValueError(
                "热导率计算需要 ther_cond_mesh 参数！\n"
                "请设置 ther_cond_mesh=[40, 40, 40]"
            )
        
        if ther_cond_temp is None or len(ther_cond_temp) < 3:
            raise ValueError(
                "热导率计算需要 ther_cond_temp 参数！\n"
                "请设置 ther_cond_temp=[300, 800, 100]  # [tmin, tmax, tstep]"
            )
        
        # ========== 分支选择：anphon vs phono3py ==========
        if anphon_ther_cond:
            logger.info("=" * 80)
            logger.info("使用 ALAMODE anphon 计算热导率（含四阶）")
            logger.info("=" * 80)
            
            # ===== 1. 保留Pheasy生成的力常数作为对比 =====
            logger.info("\n[保留Pheasy力常数文件用于对比]")
            pheasy_files = {
                "FORCE_CONSTANTS": "FORCE_CONSTANTS_pheasy",
                "FORCE_CONSTANTS_3RD": "FORCE_CONSTANTS_3RD_pheasy", 
                "FORCE_CONSTANTS_4TH": "FORCE_CONSTANTS_4TH_pheasy"
            }
            
            for src, dst in pheasy_files.items():
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    logger.info(f"  ✅ {src} → {dst}")
            
            # ===== 2. 用ALM重新拟合力常数 =====
            logger.info("\n[ALM拟合力常数（ALAMODE格式）]")
            
            # 准备ALM输入数据
            supercell_ph = phonon.supercell
            lattice = supercell_ph.cell
            positions = supercell_ph.scaled_positions
            numbers = supercell_ph.numbers
            
            # 准备DFSET文件（ALM训练数据格式）
            _prepare_alm_dfset(
                dataset_disps_array_use,
                dataset_forces_array_disp,
                num_har,
                num_anhar,
                supercell_ph
            )
            
            # 运行ALM拟合
            alm_xml_file = _run_alm_fit(
                lattice, 
                positions, 
                numbers,
                max_order=3 if cal_4th_order else 2,
                cutoffs=fcs_cutoff_radius,
                alm_save_dir=kwargs.get('alm_save_dir')
            )
            
            # ===== 3. 计算2阶声子谱（用anphon）=====
            logger.info("\n[计算2阶声子色散（anphon）]")
            
            # 获取k点路径（使用pymatgen直接获取）
            from pymatgen.symmetry.bandstructure import HighSymmKpath
            pmg_struct = get_pmg_structure(phonon.primitive)
            kpath = HighSymmKpath(pmg_struct, symprec=symprec)
            kpath_dict = kpath.kpath
            
            _run_anphon_phonons(
                structure,
                phonon.primitive,
                alm_xml_file,
                kpath_dict=kpath_dict,
                born=borns if borns is not None else None,
                epsilon=epsilon if epsilon is not None else None,
                symprec=symprec
            )
            
            # ===== 4. 计算热导率（含四阶）=====
            logger.info("\n[计算热导率（3ph+4ph）]")
            
            include_quartic = cal_4th_order
            _run_anphon_rta(
                structure,
                phonon.primitive,
                alm_xml_file,
                mesh=ther_cond_mesh,
                temp_range=ther_cond_temp,
                include_quartic=include_quartic,
                born=borns if borns is not None else None,
                epsilon=epsilon if epsilon is not None else None,
                symprec=symprec
            )
            
            logger.info("\n✅ ALAMODE anphon 计算完成")
            logger.info("   输出文件：")
            logger.info("   - alamode_phonon.bands  (声子色散)")
            logger.info("   - alamode_phonon.dos    (声子DOS)")
            logger.info("   - alamode_tc.kl         (热导率)")
            if include_quartic:
                logger.info("   - alamode_tc.kl_quartic (四阶贡献)")
            
        else:
            # ========== 原有phono3py逻辑 ==========
            logger.info("=" * 80)
            logger.info("使用 phono3py 计算热导率（仅三阶）")
            logger.info("=" * 80)
            
            fc_phonopy_text = parse_FORCE_CONSTANTS(filename="FORCE_CONSTANTS")
            write_force_constants_to_hdf5(fc_phonopy_text, filename="fc2.hdf5")
            logger.info("Generated fc2.hdf5 from harmonic force constants")

            # 修复：只有在三阶力常数文件存在时才进行热导率计算
            fc3_file = "FORCE_CONSTANTS_3RD"
            if os.path.exists(fc3_file):
                logger.info("Found FORCE_CONSTANTS_3RD, proceeding with thermal conductivity calculation")

                try:
                    # convert the 3RD order force constants to the phonopy format
                    prim_hiphive = ase_read("POSCAR")
                    supercell_hiphive = ase_read("SPOSCAR")
                    fcs = HiPhiveForceConstants.read_shengBTE(
                        supercell_hiphive, fc3_file, prim_hiphive
                    )
                    fcs.write_to_phono3py("fc3.hdf5")

                    phono3py_cmd = (
                        f"phono3py --dim {int(supercell_matrix[0][0])} "
                        f"{int(supercell_matrix[1][1])} {int(supercell_matrix[2][2])} "
                        f"--fc2 --fc3 --br --isotope --wigner "
                        f"--mesh {ther_cond_mesh[0]} {ther_cond_mesh[1]} {ther_cond_mesh[2]} "
                        f"--tmin {ther_cond_temp[0]} --tmax {ther_cond_temp[1]} "
                        f"--tstep {ther_cond_temp[2]}"
                    )

                    subprocess.call(shlex.split(phono3py_cmd))
                    logger.info("Thermal conductivity calculation completed successfully")

                except Exception as e:
                    logger.warning(f"Failed to process third-order force constants: {e}")
                    logger.warning("Thermal conductivity calculation will be skipped")
            else:
                logger.warning(f"三阶力常数文件 {fc3_file} 不存在，跳过热导率计算")
                logger.warning("请确保 cal_3rd_order=True 并提供了足够的非谐位移")

    # Read the force constants from the output file of pheasy code
    force_constants = parse_FORCE_CONSTANTS(filename=fc_file)
    phonon.force_constants = force_constants
    # symmetrize the force constants to make them physically correct based on
    # the space group symmetry of the crystal structure.
    phonon.symmetrize_force_constants()

    # with phonopy.load("phonopy.yaml") the phonopy API can be used
    phonon.save(_DEFAULT_FILE_PATHS["phonopy"])

    # get phonon band structure
    kpath_dict, kpath_concrete = _get_kpath(
        structure=get_pmg_structure(phonon.primitive),
        kpath_scheme=kpath_scheme,
        symprec=symprec,
    )

    npoints_band = kwargs.get("npoints_band", 101)
    qpoints, connections = get_band_qpoints_and_path_connections(
        kpath_concrete, npoints=npoints_band
    )

    phonon.run_band_structure(
        qpoints,
        path_connections=connections,
        with_eigenvectors=kwargs.get("band_structure_eigenvectors", False),
        is_band_connection=kwargs.get("band_structure_eigenvectors", False),
    )
    # phonon.write_hdf5_band_structure(filename=_DEFAULT_FILE_PATHS["band_structure"])
    phonon.write_yaml_band_structure(filename=_DEFAULT_FILE_PATHS["band_structure"])
    bs_symm_line = get_ph_bs_symm_line(
        _DEFAULT_FILE_PATHS["band_structure"],
        labels_dict=kpath_dict,
        has_nac=born is not None,
    )

    bs_plot_file = kwargs.get("filename_bs", _DEFAULT_FILE_PATHS["band_structure_plot"])
    dos_plot_file = kwargs.get("filename_dos", _DEFAULT_FILE_PATHS["dos_plot"])

    new_plotter = PhononBSPlotter(bs=bs_symm_line)
    new_plotter.save_plot(
        filename=bs_plot_file,
        units=kwargs.get("units", "THz"),
    )

    # will determine if imaginary modes are present in the structure
    imaginary_modes = bs_symm_line.has_imaginary_freq(
        tol=kwargs.get("tol_imaginary_modes", 1e-5)
    )

    # If imaginary modes are present, we first use the hiphive code to enforce
    # some symmetry constraints to eliminate the imaginary modes (generally work
    # for small imaginary modes near Gamma point). If the imaginary modes are
    # still present, we will use the pheasy code to generate the force constants
    # using a shorter cutoff (10 A) to eliminate the imaginary modes, also we
    # just want to remove the imaginary modes near Gamma point. In the future,
    # we will only use the pheasy code to do the job.

    if imaginary_modes:
        # Define a cluster space using the largest cutoff you can
        max_cutoff = estimate_maximum_cutoff(supercell) - 0.01
        cutoffs = [max_cutoff]  # only second order needed
        cs = ClusterSpace(prim, cutoffs)

        # import the phonopy force constants using the correct supercell also
        # provided by phonopy
        fcs = HiPhiveForceConstants.read_phonopy(supercell, "FORCE_CONSTANTS")

        # Find the parameters that best fits the force constants given you
        # cluster space
        parameters = extract_parameters(fcs, cs)

        # Enforce the rotational sum rules
        parameters_rot = enforce_rotational_sum_rules(
            cs, parameters, ["Huang", "Born-Huang"], alpha=1e-6
        )

        # use the new parameters to make a fcp and then create the force
        # constants and write to a phonopy file
        fcp = ForceConstantPotential(cs, parameters_rot)
        fcs = fcp.get_force_constants(supercell)
        new_fc_file_hiphive = f"{_DEFAULT_FILE_PATHS['force_constants']}_hiphive"
        fcs.write_to_phonopy(new_fc_file_hiphive, format="text")

        force_constants = parse_FORCE_CONSTANTS(filename=new_fc_file_hiphive)
        phonon.force_constants = force_constants
        phonon.symmetrize_force_constants()

        # ✅ 重新计算 k-path（更安全）
        kpath_dict, kpath_concrete = _get_kpath(
            structure=get_pmg_structure(phonon.primitive),
            kpath_scheme=kpath_scheme,
            symprec=symprec,
        )
        
        qpoints, connections = get_band_qpoints_and_path_connections(
            kpath_concrete, npoints=npoints_band  # ← 使用已定义的变量
        )

        phonon.run_band_structure(
            qpoints, path_connections=connections, with_eigenvectors=True
        )
        phonon.write_yaml_band_structure(filename=_DEFAULT_FILE_PATHS["band_structure"])
        bs_symm_line = get_ph_bs_symm_line(
            _DEFAULT_FILE_PATHS["band_structure"],
            labels_dict=kpath_dict,
            has_nac=born is not None,
        )

        new_plotter = PhononBSPlotter(bs=bs_symm_line)

        new_plotter.save_plot(
            filename=bs_plot_file,
            units=kwargs.get("units", "THz"),
        )

        imaginary_modes = bs_symm_line.has_imaginary_freq(
            tol=kwargs.get("tol_imaginary_modes", 1e-5)
        )

    # Using a shorter cutoff (10 A) to generate the force constants to
    # eliminate the imaginary modes near Gamma point in phesay code
    if imaginary_modes:
        logger.warning("=" * 80)
        logger.warning("检测到虚频，尝试使用更短截断重新拟合")
        
        # ========== 使用安全截断（自动估算）==========
        # 使用 hiphive 的自动估算
        max_cutoff_pheasy = estimate_maximum_cutoff(supercell) - 0.01
        
        # 如果用户明确设置了截断，使用用户的设置
        # 但要确保不超过安全范围
        if fcs_cutoff_radius and fcs_cutoff_radius[0] is not None and fcs_cutoff_radius[0] > 0:
            user_cutoff = fcs_cutoff_radius[0] * BOHR_TO_ANGSTROM
            # 使用两者中较小的值（更保守）
            reimag_cutoff = min(user_cutoff, max_cutoff_pheasy)
            logger.info(f"虚频修正截断: {reimag_cutoff:.3f} Å (用户: {user_cutoff:.3f}, 自动: {max_cutoff_pheasy:.3f})")
        else:
            # 用户未设置，使用自动估算
            reimag_cutoff = max_cutoff_pheasy
            logger.info(f"虚频修正截断: {reimag_cutoff:.3f} Å (自动估算)")
        
        # ========== 构建命令（使用计算出的截断）==========
        pheasy_cmd_11 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} -s -w 2 --c2 "
            f"{reimag_cutoff:.6f} --symprec {float(symprec)} "
            f"--nbody 2"
        )

        pheasy_cmd_12 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} -c --symprec "
            f"{float(symprec)} --c2 {reimag_cutoff:.6f} -w 2"
        )

        pheasy_cmd_13 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} -w 2 -d --symprec "
            f"{float(symprec)} --c2 {reimag_cutoff:.6f} "
            f"--ndata {int(num_har)} --disp_file"
        )

        phonon.generate_displacements(distance=displacement)

        if len(phonon.displacements) > 3:
            pheasy_cmd_14 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -f --c2 {reimag_cutoff:.6f} "
                f"--full_ifc -w 2 --symprec {float(symprec)} "
                f"-l LASSO --std --rasr BHH --ndata {int(num_har)}"
            )

        else:
            pheasy_cmd_14 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -f --full_ifc "
                f"--c2 {reimag_cutoff:.6f} -w 2 --symprec {float(symprec)} "
                f"--rasr BHH --ndata {int(num_har)}"
            )

        subprocess.call(shlex.split(pheasy_cmd_11))
        subprocess.call(shlex.split(pheasy_cmd_12))
        subprocess.call(shlex.split(pheasy_cmd_13))
        subprocess.call(shlex.split(pheasy_cmd_14))
        
        logger.info(f"虚频修正完成，使用截断: {reimag_cutoff:.3f} Å")

        fc_file = _DEFAULT_FILE_PATHS["force_constants"]  # "FORCE_CONSTANTS"
        force_constants = parse_FORCE_CONSTANTS(filename=fc_file)
        phonon.force_constants = force_constants
        phonon.symmetrize_force_constants()

        phonon.save(_DEFAULT_FILE_PATHS["phonopy"])

        # get phonon band structure
        kpath_dict, kpath_concrete = _get_kpath(
            structure=get_pmg_structure(phonon.primitive),
            kpath_scheme=kpath_scheme,
            symprec=symprec,
        )
        
        qpoints, connections = get_band_qpoints_and_path_connections(
            kpath_concrete, npoints=npoints_band
        )

        # phonon band structures will always be computed
        phonon.run_band_structure(
            qpoints, path_connections=connections, with_eigenvectors=True
        )
        phonon.write_yaml_band_structure(filename=_DEFAULT_FILE_PATHS["band_structure"])
        bs_symm_line = get_ph_bs_symm_line(
            _DEFAULT_FILE_PATHS["band_structure"],
            labels_dict=kpath_dict,
            has_nac=born is not None,
        )
        new_plotter = PhononBSPlotter(bs=bs_symm_line)

        new_plotter.save_plot(
            filename=bs_plot_file,
            units=kwargs.get("units", "THz"),
        )

        imaginary_modes = bs_symm_line.has_imaginary_freq(
            tol=kwargs.get("tol_imaginary_modes", 1e-5)
        )

    # gets data for visualization on website - yaml is also enough
    if kwargs.get("band_structure_eigenvectors"):
        bs_symm_line.write_phononwebsite(_DEFAULT_FILE_PATHS["website"])

    # get phonon density of states
    kpoint_density_dos = kwargs.get("kpoint_density_dos", 7_000)
    kpoint = Kpoints.automatic_density(
        structure=get_pmg_structure(phonon.primitive),
        kppa=kpoint_density_dos,
        force_gamma=True,
    )
    phonon.run_mesh(kpoint.kpts[0])
    phonon.run_total_dos()
    phonon.write_total_dos(filename=_DEFAULT_FILE_PATHS["dos"])
    dos = get_ph_dos(_DEFAULT_FILE_PATHS["dos"])
    new_plotter_dos = PhononDosPlotter()
    new_plotter_dos.add_dos(label="total", dos=dos)
    new_plotter_dos.save_plot(
        filename=dos_plot_file,
        units=kwargs.get("units", "THz"),
    )

    # will compute thermal displacement matrices
    # for the primitive cell (phonon.primitive!)
    # only this is available in phonopy
    if kwargs.get("create_thermal_displacements"):
        phonon.run_mesh(kpoint.kpts[0], with_eigenvectors=True, is_mesh_symmetry=False)
        freq_min_thermal_displacements = kwargs.get(
            "freq_min_thermal_displacements", 0.0
        )
        phonon.run_thermal_displacement_matrices(
            t_min=kwargs.get("tmin_thermal_displacements", 0),
            t_max=kwargs.get("tmax_thermal_displacements", 500),
            t_step=kwargs.get("tstep_thermal_displacements", 100),
            freq_min=freq_min_thermal_displacements,
        )

        temperature_range_thermal_displacements = np.arange(
            kwargs.get("tmin_thermal_displacements", 0),
            kwargs.get("tmax_thermal_displacements", 500),
            kwargs.get("tstep_thermal_displacements", 100),
        )
        for idx, temp in enumerate(temperature_range_thermal_displacements):
            phonon.thermal_displacement_matrices.write_cif(
                phonon.primitive, idx, filename=f"tdispmat_{temp}K.cif"
            )
        _disp_mat = phonon._thermal_displacement_matrices  # noqa: SLF001
        tdisp_mat = _disp_mat.thermal_displacement_matrices.tolist()

        tdisp_mat_cif = _disp_mat.thermal_displacement_matrices_cif.tolist()

    else:
        tdisp_mat = None
        tdisp_mat_cif = None

    formula_units = (
        structure.composition.num_atoms
        / structure.composition.reduced_composition.num_atoms
    )

    total_dft_energy_per_formula_unit = (
        total_dft_energy / formula_units if total_dft_energy is not None else None
    )

    cls_constructor = (
        "migrate_fields"
        if parse_version(_emmet_core_version) >= parse_version("0.85.1")
        else "from_structure"
    )

    # 先将 PhononBSDOSDoc 转换为字典，再处理复数
    output_data = getattr(PhononBSDOSDoc, cls_constructor)(
        structure=structure,
        meta_structure=structure,
        phonon_bandstructure=bs_symm_line,
        phonon_dos=dos,
        total_dft_energy=total_dft_energy_per_formula_unit,
        has_imaginary_modes=imaginary_modes,
        force_constants=(
            {"force_constants": phonon.force_constants.tolist()}
            if kwargs.get("store_force_constants")
            else None
        ),
        born=borns.tolist() if borns is not None else None,
        epsilon_static=epsilon.tolist() if epsilon is not None else None,
        supercell_matrix=phonon.supercell_matrix.tolist(),
        primitive_matrix=phonon.primitive_matrix.tolist(),
        code=code,
        thermal_displacement_data={
            "temperatures_thermal_displacements": temperature_range_thermal_displacements.tolist(),
            "thermal_displacement_matrix_cif": tdisp_mat_cif,
            "thermal_displacement_matrix": tdisp_mat,
            "freq_min_thermal_displacements": freq_min_thermal_displacements,
        }
        if kwargs.get("create_thermal_displacements")
        else None,
        jobdirs={
            "displacements_job_dirs": displacement_data["dirs"],
            "static_run_job_dir": kwargs["static_run_job_dir"],
            "born_run_job_dir": kwargs["born_run_job_dir"],
            "optimization_run_job_dir": kwargs["optimization_run_job_dir"],
            "taskdoc_run_job_dir": str(Path.cwd()),
        },
        uuids={
            "displacements_uuids": displacement_data["uuids"],
            "born_run_uuid": kwargs["born_run_uuid"],
            "optimization_run_uuid": kwargs["optimization_run_uuid"],
            "static_run_uuid": kwargs["static_run_uuid"],
        },
        post_process_settings={
            "npoints_band": npoints_band,
            "kpath_scheme": kpath_scheme,
            "kpoint_density_dos": kpoint_density_dos,
        },
    )
    # 先将结果转换为字典
    output_dict = output_data.as_dict() if hasattr(output_data, 'as_dict') else output_data
    
    logger.debug(f"Raw output_dict: {output_dict}")
    # 清理复数，确保所有数据可序列化
    output_dict_sanitized = sanitize_complex(output_dict)
    logger.debug(f"Sanitized output_dict: {output_dict_sanitized}")

    
    # 返回清理后的字典
    return output_dict_sanitized
    # 恢复原始返回：
    #return output_data


def _prepare_alm_dfset(disps, forces, n_harmonic, n_anharmonic, supercell):
    """
    准备ALM的DFSET文件（位移-力数据集）
    
    Parameters
    ----------
    disps : array (n_config, n_atoms, 3)
        位移（分数坐标）
    forces : array (n_config, n_atoms, 3)
        力（Cartesian坐标, eV/Å）
    n_harmonic : int
        谐波位移数量
    n_anharmonic : int
        非谐波位移数量
    supercell : phonopy.structure
        超胞结构
    """
    from ase.io import write as ase_write
    from ase import Atoms
    
    logger.info(f"准备ALM训练数据: {n_harmonic} 谐波 + {n_anharmonic} 非谐")
    
    # 合并谐波和非谐数据
    all_disps = disps[:n_harmonic + n_anharmonic]
    all_forces = forces[:n_harmonic + n_anharmonic]
    
    # 转换为Cartesian位移
    all_disps_cart = np.einsum('ij,njk->nik', supercell.cell.T,
                                np.transpose(all_disps, (0, 2, 1)))
    all_disps_cart = np.transpose(all_disps_cart, (0, 2, 1))
    
    # ALM单位转换常数
    ANGSTROM_TO_BOHR = 1.889726124565062  # Å -> Bohr
    EV_PER_ANG_TO_HARTREE_PER_BOHR = 0.01944690408  # eV/Å -> Hartree/Bohr
    
    # 写入DFSET（ALM格式）
    # 重要：ALM期望每个配置恰好NAT行，不要配置头和注释！
    with open("DFSET", "w") as f:
        for i_config in range(len(all_disps)):
            # 每个原子的位移和力（不要配置头！）
            for i_atom in range(len(supercell.numbers)):
                disp = all_disps_cart[i_config, i_atom]  # Å
                force = all_forces[i_config, i_atom]  # eV/Å
                
                # 单位转换
                disp_bohr = disp * ANGSTROM_TO_BOHR
                force_hartree = force * EV_PER_ANG_TO_HARTREE_PER_BOHR
                
                # ALM格式：dx dy dz fx fy fz (Bohr, Hartree/Bohr)
                f.write(f"{disp_bohr[0]:20.15f} {disp_bohr[1]:20.15f} {disp_bohr[2]:20.15f}  ")
                f.write(f"{force_hartree[0]:20.15f} {force_hartree[1]:20.15f} {force_hartree[2]:20.15f}\n")
    
    logger.info(f"✅ DFSET文件已生成 ({len(all_disps)} 配置)")


def _run_alm_fit(lattice, positions, numbers, max_order=3, cutoffs=None, alm_save_dir=None):
    """
    运行ALM拟合力常数（MODE=opt）
    
    Parameters
    ----------
    max_order : int
        最大阶数 (1=2阶, 2=2+3阶, 3=2+3+4阶)
    cutoffs : list
        截断半径 [2阶, 3阶, 4阶] (Bohr)
    
    Returns
    -------
    str
        生成的XML文件路径
    """
    import tempfile
    
    # 使用ALM类创建输入文件
    with ALM(lattice, positions, numbers, alm_save_dir=alm_save_dir) as alm:
        alm.max_order = max_order
        alm.cutoffs = cutoffs
        
        # ===== 关键修复：复制DFSET文件到ALM工作目录 =====
        import shutil
        dfset_source = os.path.join(alm.original_dir, "DFSET")
        if os.path.exists(dfset_source):
            shutil.copy2(dfset_source, "DFSET")
            logger.info(f"✅ DFSET已复制到ALM工作目录")
        else:
            raise FileNotFoundError(f"找不到DFSET文件: {dfset_source}")
        # =====================================================
        
        # 创建优化模式的输入文件
        alm._create_alm_fit_input(max_order, cutoffs)
        
        # 运行ALM
        logger.info(f"🚀 运行ALM拟合 (NORDER={max_order})")
        success = alm._run_alm()
        
        if not success:
            raise RuntimeError("ALM拟合失败！")
        
        # ===== 检查输出文件（兼容新旧版本ALM）=====
        # ALM 1.9.9+ 生成通用文件名 alm.xml
        # 旧版本生成特定阶数文件名 alm_FC2.xml, alm_FC3.xml, alm_FC4.xml
        
        # 优先检查新版本格式
        if os.path.exists("alm.xml"):
            xml_filename = "alm.xml"
            logger.info("✅ 检测到ALM 1.9.9+格式: alm.xml")
        else:
            # 回退到旧版本格式
            xml_files = {
                1: "alm_FC2.xml",
                2: "alm_FC3.xml", 
                3: "alm_FC4.xml"
            }
            xml_filename = xml_files.get(max_order, "alm_FC3.xml")
            
            if not os.path.exists(xml_filename):
                raise FileNotFoundError(
                    f"ALM未生成XML文件！\n"
                    f"  期望: alm.xml 或 {xml_filename}\n"
                    f"  工作目录: {os.getcwd()}\n"
                    f"  请检查alm.log"
                )
            logger.info(f"✅ 检测到旧版ALM格式: {xml_filename}")
        
        logger.info(f"✅ ALM拟合完成: {xml_filename}")
        
        # ===== 关键修复：确定XML文件将被保存的完整路径 =====
        # __exit__会将文件复制到save_dir
        save_dir = alm.alm_save_dir if alm.alm_save_dir else alm.original_dir
    
    # 退出with后，XML文件已被复制到save_dir，返回完整路径
    xml_file_path = os.path.join(save_dir, xml_filename)
    logger.info(f"📁 XML文件路径: {xml_file_path}")
    
    # 验证文件存在
    if not os.path.exists(xml_file_path):
        raise FileNotFoundError(f"XML文件未找到: {xml_file_path}")
    
    return xml_file_path


def _run_anphon_phonons(structure, primitive, xml_file, kpath_dict, 
                        born=None, epsilon=None, symprec=1e-5):
    """
    用anphon计算2阶声子谱
    
    Parameters
    ----------
    structure : Structure
        原始结构
    primitive : phonopy primitive
        原胞结构
    xml_file : str
        ALM生成的XML文件
    kpath_dict : dict
        k路径字典
    born : array, optional
        Born有效电荷
    epsilon : array, optional
        介电常数
    """
    logger.info("准备anphon声子计算输入...")
    
    # 生成k点路径
    kpath_lines = []
    for path in kpath_dict['path']:
        for i in range(len(path) - 1):
            start_label = path[i]
            end_label = path[i+1]
            start_coord = kpath_dict['kpoints'][start_label]
            end_coord = kpath_dict['kpoints'][end_label]
            
            kpath_lines.append(
                f"{start_label:5s} {start_coord[0]:10.6f} {start_coord[1]:10.6f} {start_coord[2]:10.6f}  "
                f"{end_label:5s} {end_coord[0]:10.6f} {end_coord[1]:10.6f} {end_coord[2]:10.6f}  51"
            )
    
    # 写入anphon输入文件
    with open("anphon_phonon.in", "w") as f:
        f.write("&general\n")
        f.write(" PREFIX = alamode_phonon\n")
        f.write(" MODE = phonons\n")
        f.write(f" FCSFILE = {xml_file}\n")  # 使用FCSFILE而不是FCSXML
        
        # NAC修正
        if born is not None and epsilon is not None:
            f.write(" NONANALYTIC = 3\n")  # Ewald方法
            f.write(" BORNINFO = BORN_anphon\n")
            
            # 写入BORN文件
            _write_born_file(primitive, born, epsilon, "BORN_anphon")
        else:
            f.write(" NONANALYTIC = 0\n")
        
        f.write("/\n\n")
        
        # &cell
        lattice_const = np.linalg.norm(primitive.cell[0])
        normalized_lattice = primitive.cell / lattice_const
        
        f.write("&cell\n")
        f.write(f" {lattice_const:.10f}\n")
        for vec in normalized_lattice:
            f.write(f" {vec[0]:20.15f} {vec[1]:20.15f} {vec[2]:20.15f}\n")
        f.write("/\n\n")
        
        # &kpoint (能带路径)
        f.write("&kpoint\n")
        f.write(" 1\n")  # KPMODE=1 (line mode)
        for line in kpath_lines:
            f.write(f" {line}\n")
        f.write("/\n")
    
    # 运行anphon
    anphon_bin = shutil.which("anphon")
    if not anphon_bin:
        raise RuntimeError(
            "anphon 可执行文件未找到！\n"
            "请确保 'anphon' 在系统 PATH 中。\n"
            "可以通过以下方式添加：\n"
            "  export PATH=/path/to/alamode/build/anphon:$PATH"
        )
    
    logger.info(f"运行: {anphon_bin} anphon_phonon.in")
    result = subprocess.call([anphon_bin, "anphon_phonon.in"], 
                            stdout=open("anphon_phonon.log", "w"),
                            stderr=subprocess.STDOUT)
    
    if result != 0:
        logger.warning(f"⚠️  anphon声子计算退出码: {result}")
    
    if os.path.exists("alamode_phonon.bands"):
        logger.info("✅ 声子色散计算完成: alamode_phonon.bands")
    else:
        logger.error("❌ 未生成声子色散文件")


def _run_anphon_rta(structure, primitive, xml_file, mesh, temp_range,
                    include_quartic=True, born=None, epsilon=None, symprec=1e-5):
    """
    用anphon计算热导率（RTA方法）
    
    Parameters
    ----------
    mesh : list
        k点网格 [nx, ny, nz]
    temp_range : list
        温度范围 [tmin, tmax, tstep]
    include_quartic : bool
        是否包含四声子散射
    """
    logger.info("准备anphon热导率计算输入...")
    
    with open("anphon_rta.in", "w") as f:
        f.write("&general\n")
        f.write(" PREFIX = alamode_tc\n")
        f.write(" MODE = RTA\n")
        f.write(f" FCSFILE = {xml_file}\n")  # 使用FCSFILE而不是FCSXML
        
        # NAC修正
        if born is not None and epsilon is not None:
            f.write(" NONANALYTIC = 3\n")
            f.write(" BORNINFO = BORN_anphon\n")
        else:
            f.write(" NONANALYTIC = 0\n")
        
        f.write("/\n\n")
        
        # &cell
        lattice_const = np.linalg.norm(primitive.cell[0])
        normalized_lattice = primitive.cell / lattice_const
        
        f.write("&cell\n")
        f.write(f" {lattice_const:.10f}\n")
        for vec in normalized_lattice:
            f.write(f" {vec[0]:20.15f} {vec[1]:20.15f} {vec[2]:20.15f}\n")
        f.write("/\n\n")
        
        # &kpoint (uniform mesh)
        f.write("&kpoint\n")
        f.write(" 2\n")  # KPMODE=2 (uniform mesh)
        f.write(f" {mesh[0]} {mesh[1]} {mesh[2]}\n")
        f.write("/\n\n")
        
        # &temperature
        f.write("&temperature\n")
        f.write(f" TMIN = {temp_range[0]}; TMAX = {temp_range[1]}; DT = {temp_range[2]}\n")
        f.write("/\n\n")
        
        # &anharmonic (四阶开关)
        f.write("&anharmonic\n")
        if include_quartic:
            f.write(" QUARTIC = 1\n")
            logger.info("  ✅ 四声子散射: 开启")
        else:
            f.write(" QUARTIC = 0\n")
            logger.info("  ℹ️  四声子散射: 关闭")
        f.write("/\n")
    
    # 运行anphon（暂时禁用MPI以避免配置问题）
    anphon_bin = shutil.which("anphon")
    if not anphon_bin:
        raise RuntimeError(
            "anphon 可执行文件未找到！\n"
            "请确保 'anphon' 在系统 PATH 中。\n"
            "可以通过以下方式添加：\n"
            "  export PATH=/path/to/alamode/build/anphon:$PATH"
        )
    
    # 直接运行，不使用MPI（避免SLURM PMI配置问题）
    cmd = f"{anphon_bin} anphon_rta.in"
    
    logger.info(f"运行: {cmd}")
    logger.info("  ℹ️  注意：当前禁用MPI并行以避免配置问题")
    result = subprocess.call(cmd, shell=True,
                            stdout=open("anphon_rta.log", "w"),
                            stderr=subprocess.STDOUT)
    
    if result != 0:
        logger.warning(f"⚠️  anphon热导率计算退出码: {result}")
    
    # 检查输出
    if os.path.exists("alamode_tc.kl"):
        logger.info("✅ 热导率计算完成: alamode_tc.kl")
    
    if include_quartic and os.path.exists("alamode_tc.kl_quartic"):
        logger.info("✅ 四阶贡献文件: alamode_tc.kl_quartic")


def _write_born_file(primitive, born, epsilon, filename):
    """写入ALAMODE格式的BORN文件"""
    with open(filename, "w") as f:
        # 介电常数张量
        f.write("# Dielectric constant\n")
        for row in epsilon:
            f.write(f"{row[0]:15.8f} {row[1]:15.8f} {row[2]:15.8f}\n")
        
        f.write("\n# Born effective charges\n")
        for i_atom, Z_tensor in enumerate(born):
            f.write(f"# Atom {i_atom+1}\n")
            for row in Z_tensor:
                f.write(f"{row[0]:15.8f} {row[1]:15.8f} {row[2]:15.8f}\n")
