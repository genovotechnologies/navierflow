import unittest
import numpy as np
import torch
from navierflow.core.physics.fluid import FluidFlow
from navierflow.core.physics.turbulence import TurbulenceModel
from navierflow.core.physics.multiphase import MultiphaseFlow
from navierflow.core.physics.electromagnetic import ElectromagneticField
from navierflow.core.physics.non_newtonian import NonNewtonianFlow
from navierflow.core.numerics.solver import Solver
from navierflow.core.numerics.boundary import BoundaryManager
from navierflow.core.mesh.generation import MeshGenerator

class TestFluidFlow(unittest.TestCase):
    def setUp(self):
        """Set up test case"""
        self.fluid = FluidFlow(
            density=1.0,
            viscosity=0.1,
            gravity=(0.0, -9.81, 0.0)
        )
        
    def test_compute_pressure(self):
        """Test pressure computation"""
        # Create test data
        velocity = np.zeros((10, 10, 10, 3))
        velocity[..., 0] = 1.0
        
        # Compute pressure
        pressure = self.fluid.compute_pressure(velocity)
        
        # Check pressure shape
        self.assertEqual(pressure.shape, (10, 10, 10))
        
        # Check pressure values
        self.assertTrue(np.all(pressure >= 0))
        
    def test_compute_vorticity(self):
        """Test vorticity computation"""
        # Create test data
        velocity = np.zeros((10, 10, 10, 3))
        velocity[..., 0] = 1.0
        
        # Compute vorticity
        vorticity = self.fluid.compute_vorticity(velocity)
        
        # Check vorticity shape
        self.assertEqual(vorticity.shape, (10, 10, 10, 3))
        
    def test_compute_strain_rate(self):
        """Test strain rate computation"""
        # Create test data
        velocity = np.zeros((10, 10, 10, 3))
        velocity[..., 0] = 1.0
        
        # Compute strain rate
        strain_rate = self.fluid.compute_strain_rate(velocity)
        
        # Check strain rate shape
        self.assertEqual(strain_rate.shape, (10, 10, 10, 3, 3))
        
    def test_compute_energy(self):
        """Test energy computation"""
        # Create test data
        velocity = np.zeros((10, 10, 10, 3))
        velocity[..., 0] = 1.0
        
        # Compute energy
        energy = self.fluid.compute_energy(velocity)
        
        # Check energy shape
        self.assertEqual(energy.shape, (10, 10, 10))
        
        # Check energy values
        self.assertTrue(np.all(energy >= 0))

class TestTurbulenceModel(unittest.TestCase):
    def setUp(self):
        """Set up test case"""
        self.turbulence = TurbulenceModel(
            model_type="k_epsilon",
            c_mu=0.09,
            c_epsilon1=1.44,
            c_epsilon2=1.92,
            sigma_k=1.0,
            sigma_epsilon=1.3
        )
        
    def test_compute_eddy_viscosity(self):
        """Test eddy viscosity computation"""
        # Create test data
        velocity = np.zeros((10, 10, 10, 3))
        velocity[..., 0] = 1.0
        
        # Compute eddy viscosity
        eddy_viscosity = self.turbulence.compute_eddy_viscosity(velocity)
        
        # Check eddy viscosity shape
        self.assertEqual(eddy_viscosity.shape, (10, 10, 10))
        
        # Check eddy viscosity values
        self.assertTrue(np.all(eddy_viscosity >= 0))
        
    def test_compute_turbulent_kinetic_energy(self):
        """Test turbulent kinetic energy computation"""
        # Create test data
        velocity = np.zeros((10, 10, 10, 3))
        velocity[..., 0] = 1.0
        
        # Compute turbulent kinetic energy
        k = self.turbulence.compute_turbulent_kinetic_energy(velocity)
        
        # Check turbulent kinetic energy shape
        self.assertEqual(k.shape, (10, 10, 10))
        
        # Check turbulent kinetic energy values
        self.assertTrue(np.all(k >= 0))
        
    def test_compute_dissipation_rate(self):
        """Test dissipation rate computation"""
        # Create test data
        velocity = np.zeros((10, 10, 10, 3))
        velocity[..., 0] = 1.0
        
        # Compute dissipation rate
        epsilon = self.turbulence.compute_dissipation_rate(velocity)
        
        # Check dissipation rate shape
        self.assertEqual(epsilon.shape, (10, 10, 10))
        
        # Check dissipation rate values
        self.assertTrue(np.all(epsilon >= 0))

class TestMultiphaseFlow(unittest.TestCase):
    def setUp(self):
        """Set up test case"""
        self.multiphase = MultiphaseFlow(
            phases=["water", "air"],
            densities=[1000.0, 1.0],
            viscosities=[0.001, 0.00001],
            surface_tension=0.072
        )
        
    def test_compute_volume_fraction(self):
        """Test volume fraction computation"""
        # Create test data
        level_set = np.zeros((10, 10, 10))
        level_set[5:, :, :] = 1.0
        
        # Compute volume fraction
        volume_fraction = self.multiphase.compute_volume_fraction(level_set)
        
        # Check volume fraction shape
        self.assertEqual(volume_fraction.shape, (10, 10, 10))
        
        # Check volume fraction values
        self.assertTrue(np.all(volume_fraction >= 0))
        self.assertTrue(np.all(volume_fraction <= 1))
        
    def test_compute_interface_normal(self):
        """Test interface normal computation"""
        # Create test data
        level_set = np.zeros((10, 10, 10))
        level_set[5:, :, :] = 1.0
        
        # Compute interface normal
        normal = self.multiphase.compute_interface_normal(level_set)
        
        # Check interface normal shape
        self.assertEqual(normal.shape, (10, 10, 10, 3))
        
        # Check interface normal values
        self.assertTrue(np.all(np.abs(normal) <= 1))
        
    def test_compute_curvature(self):
        """Test curvature computation"""
        # Create test data
        level_set = np.zeros((10, 10, 10))
        level_set[5:, :, :] = 1.0
        
        # Compute curvature
        curvature = self.multiphase.compute_curvature(level_set)
        
        # Check curvature shape
        self.assertEqual(curvature.shape, (10, 10, 10))
        
    def test_compute_surface_tension(self):
        """Test surface tension computation"""
        # Create test data
        level_set = np.zeros((10, 10, 10))
        level_set[5:, :, :] = 1.0
        
        # Compute surface tension
        surface_tension = self.multiphase.compute_surface_tension(level_set)
        
        # Check surface tension shape
        self.assertEqual(surface_tension.shape, (10, 10, 10, 3))

class TestElectromagneticField(unittest.TestCase):
    def setUp(self):
        """Set up test case"""
        self.em = ElectromagneticField(
            permittivity=8.85e-12,
            permeability=1.26e-6,
            conductivity=1.0
        )
        
    def test_compute_electric_field(self):
        """Test electric field computation"""
        # Create test data
        charge_density = np.zeros((10, 10, 10))
        charge_density[5, 5, 5] = 1.0
        
        # Compute electric field
        electric_field = self.em.compute_electric_field(charge_density)
        
        # Check electric field shape
        self.assertEqual(electric_field.shape, (10, 10, 10, 3))
        
    def test_compute_magnetic_field(self):
        """Test magnetic field computation"""
        # Create test data
        current_density = np.zeros((10, 10, 10, 3))
        current_density[..., 0] = 1.0
        
        # Compute magnetic field
        magnetic_field = self.em.compute_magnetic_field(current_density)
        
        # Check magnetic field shape
        self.assertEqual(magnetic_field.shape, (10, 10, 10, 3))
        
    def test_compute_lorentz_force(self):
        """Test Lorentz force computation"""
        # Create test data
        electric_field = np.zeros((10, 10, 10, 3))
        magnetic_field = np.zeros((10, 10, 10, 3))
        electric_field[..., 0] = 1.0
        magnetic_field[..., 1] = 1.0
        
        # Compute Lorentz force
        lorentz_force = self.em.compute_lorentz_force(
            electric_field,
            magnetic_field
        )
        
        # Check Lorentz force shape
        self.assertEqual(lorentz_force.shape, (10, 10, 10, 3))
        
    def test_compute_boundary_conditions(self):
        """Test boundary conditions computation"""
        # Create test data
        electric_field = np.zeros((10, 10, 10, 3))
        magnetic_field = np.zeros((10, 10, 10, 3))
        
        # Compute boundary conditions
        self.em.compute_boundary_conditions(
            electric_field,
            magnetic_field,
            boundary_type="periodic"
        )
        
        # Check boundary conditions
        self.assertTrue(np.allclose(
            electric_field[0],
            electric_field[-1]
        ))
        self.assertTrue(np.allclose(
            magnetic_field[0],
            magnetic_field[-1]
        ))

class TestNonNewtonianFlow(unittest.TestCase):
    def setUp(self):
        """Set up test case"""
        self.non_newtonian = NonNewtonianFlow(
            model_type="power_law",
            k=1.0,
            n=0.5
        )
        
    def test_compute_viscosity(self):
        """Test viscosity computation"""
        # Create test data
        strain_rate = np.zeros((10, 10, 10, 3, 3))
        strain_rate[..., 0, 0] = 1.0
        
        # Compute viscosity
        viscosity = self.non_newtonian.compute_viscosity(strain_rate)
        
        # Check viscosity shape
        self.assertEqual(viscosity.shape, (10, 10, 10))
        
        # Check viscosity values
        self.assertTrue(np.all(viscosity >= 0))
        
    def test_compute_stress(self):
        """Test stress computation"""
        # Create test data
        strain_rate = np.zeros((10, 10, 10, 3, 3))
        strain_rate[..., 0, 0] = 1.0
        
        # Compute stress
        stress = self.non_newtonian.compute_stress(strain_rate)
        
        # Check stress shape
        self.assertEqual(stress.shape, (10, 10, 10, 3, 3))
        
    def test_compute_yield_criterion(self):
        """Test yield criterion computation"""
        # Create test data
        stress = np.zeros((10, 10, 10, 3, 3))
        stress[..., 0, 0] = 1.0
        
        # Compute yield criterion
        yield_criterion = self.non_newtonian.compute_yield_criterion(stress)
        
        # Check yield criterion shape
        self.assertEqual(yield_criterion.shape, (10, 10, 10))
        
    def test_compute_power_dissipation(self):
        """Test power dissipation computation"""
        # Create test data
        strain_rate = np.zeros((10, 10, 10, 3, 3))
        stress = np.zeros((10, 10, 10, 3, 3))
        strain_rate[..., 0, 0] = 1.0
        stress[..., 0, 0] = 1.0
        
        # Compute power dissipation
        power_dissipation = self.non_newtonian.compute_power_dissipation(
            strain_rate,
            stress
        )
        
        # Check power dissipation shape
        self.assertEqual(power_dissipation.shape, (10, 10, 10))
        
        # Check power dissipation values
        self.assertTrue(np.all(power_dissipation >= 0))

class TestSolver(unittest.TestCase):
    def setUp(self):
        """Set up test case"""
        self.solver = Solver(
            method="explicit",
            time_step=0.001,
            max_steps=1000,
            tolerance=1e-6
        )
        
    def test_solve(self):
        """Test solver"""
        # Create test data
        initial_condition = np.zeros((10, 10, 10, 3))
        initial_condition[..., 0] = 1.0
        
        # Solve
        solution = self.solver.solve(initial_condition)
        
        # Check solution shape
        self.assertEqual(solution.shape, (10, 10, 10, 3))
        
    def test_compute_residual(self):
        """Test residual computation"""
        # Create test data
        solution = np.zeros((10, 10, 10, 3))
        solution[..., 0] = 1.0
        
        # Compute residual
        residual = self.solver.compute_residual(solution)
        
        # Check residual shape
        self.assertEqual(residual.shape, (10, 10, 10, 3))
        
    def test_check_convergence(self):
        """Test convergence check"""
        # Create test data
        residual = np.zeros((10, 10, 10, 3))
        residual[..., 0] = 1e-7
        
        # Check convergence
        converged = self.solver.check_convergence(residual)
        
        # Check convergence
        self.assertTrue(converged)

class TestBoundaryManager(unittest.TestCase):
    def setUp(self):
        """Set up test case"""
        self.boundary = BoundaryManager()
        
    def test_add_boundary_condition(self):
        """Test boundary condition addition"""
        # Add boundary condition
        self.boundary.add_boundary_condition(
            "wall",
            "no_slip",
            value=0.0
        )
        
        # Check boundary condition
        self.assertEqual(len(self.boundary.boundary_conditions), 1)
        
    def test_apply_boundary_conditions(self):
        """Test boundary condition application"""
        # Create test data
        field = np.zeros((10, 10, 10, 3))
        field[..., 0] = 1.0
        
        # Add boundary condition
        self.boundary.add_boundary_condition(
            "wall",
            "no_slip",
            value=0.0
        )
        
        # Apply boundary conditions
        self.boundary.apply_boundary_conditions(field)
        
        # Check boundary conditions
        self.assertTrue(np.allclose(field[0], 0.0))
        self.assertTrue(np.allclose(field[-1], 0.0))
        
    def test_compute_boundary_flux(self):
        """Test boundary flux computation"""
        # Create test data
        field = np.zeros((10, 10, 10, 3))
        field[..., 0] = 1.0
        
        # Add boundary condition
        self.boundary.add_boundary_condition(
            "wall",
            "no_slip",
            value=0.0
        )
        
        # Compute boundary flux
        flux = self.boundary.compute_boundary_flux(field)
        
        # Check boundary flux shape
        self.assertEqual(flux.shape, (10, 10, 10, 3))
        
    def test_compute_boundary_forces(self):
        """Test boundary force computation"""
        # Create test data
        stress = np.zeros((10, 10, 10, 3, 3))
        stress[..., 0, 0] = 1.0
        
        # Add boundary condition
        self.boundary.add_boundary_condition(
            "wall",
            "no_slip",
            value=0.0
        )
        
        # Compute boundary forces
        forces = self.boundary.compute_boundary_forces(stress)
        
        # Check boundary forces shape
        self.assertEqual(forces.shape, (10, 10, 10, 3))

class TestMeshGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test case"""
        self.mesh = MeshGenerator(
            mesh_type="structured",
            dimension=3,
            resolution=(10, 10, 10)
        )
        
    def test_generate_structured_mesh(self):
        """Test structured mesh generation"""
        # Generate mesh
        mesh = self.mesh.generate_structured_mesh(
            bounds=((0, 0, 0), (1, 1, 1)),
            periodic=(False, False, False)
        )
        
        # Check mesh
        self.assertEqual(mesh.shape, (10, 10, 10, 3))
        
    def test_generate_unstructured_mesh(self):
        """Test unstructured mesh generation"""
        # Generate mesh
        mesh = self.mesh.generate_unstructured_mesh(
            points=np.random.rand(100, 3),
            boundary_points=np.random.rand(20, 3)
        )
        
        # Check mesh
        self.assertTrue(hasattr(mesh, "points"))
        self.assertTrue(hasattr(mesh, "cells"))
        
    def test_generate_adaptive_mesh(self):
        """Test adaptive mesh generation"""
        # Generate mesh
        mesh = self.mesh.generate_adaptive_mesh(
            initial_mesh=self.mesh.generate_structured_mesh(
                bounds=((0, 0, 0), (1, 1, 1)),
                periodic=(False, False, False)
            ),
            error_indicator=np.random.rand(10, 10, 10),
            max_refinement_level=3
        )
        
        # Check mesh
        self.assertTrue(hasattr(mesh, "points"))
        self.assertTrue(hasattr(mesh, "cells"))
        
    def test_generate_curved_mesh(self):
        """Test curved mesh generation"""
        # Generate mesh
        mesh = self.mesh.generate_curved_mesh(
            base_mesh=self.mesh.generate_structured_mesh(
                bounds=((0, 0, 0), (1, 1, 1)),
                periodic=(False, False, False)
            ),
            boundary_layers=2,
            growth_rate=1.2
        )
        
        # Check mesh
        self.assertTrue(hasattr(mesh, "points"))
        self.assertTrue(hasattr(mesh, "cells"))

if __name__ == "__main__":
    unittest.main() 