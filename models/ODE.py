import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.parametrizations import orthogonal
from models.Gelu import Gelu
from models.TimeEmbedding import TimeEmbedding
from models.Differentiate_new import Differentiate
import math


def ode_solve(z0, u, t0, t1, f):
    z = z0
    dz = f(z, u, t0, t1)
    z = z + dz
    # dt = t1 - t0
    # K1, _, _ = f(z, u, x_shape, t0, t1)
    # K2, _, _ = f(z + K1 / 2, u, x_shape, t0 + dt / 2, t1)
    # K3, _, _ = f(z + K2 / 2, u, x_shape, t0 + dt / 2, t1)
    # z = z + (K1 + 2 * K2 + 2 * K3) / 5

    return z

class ODEF(nn.Module):
    def __init__(self, time_embedding):
        super(ODEF, self).__init__()
        self.time_embedding = time_embedding
    def forward_with_grad(self, z, u, t0, t1, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]
        out = self.forward(z, u, t0, t1)
        a = grad_outputs
        adfdz, adfdu, adfdt0, adfdt1= torch.autograd.grad(
            (out,), (z, u, t0, t1), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back
        adfdp = []
        for param in self.parameters():
            grad = torch.autograd.grad(
                (out,), (param,), grad_outputs=(a),
                allow_unused=True, retain_graph=True
            )
            if grad[0] is None:
                grad = (torch.zeros_like(param),)
            adfdp.append(grad[0])

        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        return out, adfdz, adfdu, adfdt0, adfdt1, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)


class GraphODEF(ODEF):
    def __init__(self, time_embedding, graph_size, embed, device):
        super(GraphODEF, self).__init__(time_embedding)
        #  max_step, hid_dim, hid_graph_num, hid_graph_size, device
        self.graph_size = graph_size
        self.differentiate = Differentiate(embed)
        self.embed = embed

        self.n_dim_A = 3 * graph_size * graph_size
        self.n_dim_x = 3 * graph_size * embed
        self.n_dim_u = 3 * embed

        self.device = device

    def forward(self, z, t0, t1):

        A = z[:, :self.n_dim_A]
        x = z[:, self.n_dim_A:self.n_dim_A + self.n_dim_x]
        u = z[:, self.n_dim_A + self.n_dim_x:]

        A = A.view(-1, 3, self.graph_size, self.graph_size)
        x = x.view(-1, 3, self.graph_size, self.embed)
        u = u.view(-1, 3, self.embed)

        dA, dx, du = self.differentiate(A, x, u, t0, t1)

        dA = dA.view(-1, self.n_dim_A)
        dx = dx.view(-1, self.n_dim_x)
        du = du.view(-1, self.n_dim_u)

        dz = torch.cat([dA, dx, du], dim=-1).nan_to_num()

        assert not torch.isnan(dz).any()

        return dz

class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, u, t0, t1, flat_parameters, func):
        assert isinstance(func, ODEF)

        bs, *z_shape = z0.size()

        time_len = t0.size(0)
        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            for i_t in range(time_len):
                z0 = ode_solve(z0, u, t0[i_t], t1[i_t], func)
                z[i_t] = z0
        ctx.func = func
        ctx.save_for_backward(z.clone(), u.clone(), t0.clone(), t1.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        z, u, t0, t1, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        _, *u_shape = u.size()
        _, _, *t_shape = t0.size()
        n_dim_z = np.prod(z_shape)
        n_dim_u = np.prod(u_shape)
        n_dim_t = np.prod(t_shape)
        n_params = flat_parameters.size(0)
        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, u, t_i, t_i_1):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
            """
            z, a = aug_z_i[:, :n_dim_z], aug_z_i[:, n_dim_z:2 * n_dim_z]  # ignore parameters and time

            # Unflatten z and a
            z = z.view(bs, *z_shape)
            a = z.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                t_i_1 = t_i_1.detach().requires_grad_(True)
                u = u.detach().requires_grad_(True)
                z_i = z.detach().requires_grad_(True)
                func_eval, adfdz, adfdu, adfdt0, adfdt1, adfdp = \
                    func.forward_with_grad(z_i, u, t_i, t_i_1, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdu = adfdu.to(z_i) if adfdu is not None else torch.zeros(bs, *u_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt0 = adfdt0.to(z_i) if adfdt0 is not None else torch.zeros(bs, *t_shape).to(z_i)
                adfdt1 = adfdt1.to(z_i) if adfdt1 is not None else torch.zeros(bs, *t_shape).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.reshape(bs, n_dim_z)
            adfdz = adfdz.reshape(bs, n_dim_z)
            adfdu = adfdu.reshape(bs, n_dim_u)
            adfdt0 = adfdt0.reshape(bs, n_dim_t)
            adfdt1 = adfdt1.reshape(bs, n_dim_t)
            aug_z_i_1 = torch.cat((func_eval, -adfdz, -adfdu, -adfdp, -adfdt0, -adfdt1), dim=1)
            return aug_z_i_1

        dLdz = dLdz.view(time_len, bs, n_dim_z)  # flatten dLdz for convenience
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                ## Create placeholders for output gradients
                # Prev computed backwards adjoints to be adjusted by direct gradients
                adj_z = torch.zeros(bs, n_dim_z).to(dLdz)
                adj_u = torch.zeros(bs, n_dim_u).to(dLdz)
                adj_p = torch.zeros(bs, n_params).to(dLdz)
                # In contrast to z and p we need to return gradients for all times
                adj_t0 = torch.zeros(time_len, bs, n_dim_t).to(dLdz)
                adj_t1 = torch.zeros(time_len, bs, n_dim_t).to(dLdz)
                for i_t in range(time_len - 1, 0, -1):
                    z_i = z[i_t]
                    u_i = u
                    t0_i = t0[i_t]
                    t1_i = t1[i_t]
                    dLdz_i = dLdz[i_t]
                    dLdx_i = dLdz_i.view(bs, *z_shape)[:, :, :, func.graph_size:]
                    dLdx_i = dLdx_i[:, :, -1, :]
                    with torch.set_grad_enabled(True):
                        u_i = u_i.detach().requires_grad_(True)
                        t0_i = t0_i.detach().requires_grad_(True)
                        t1_i = t1_i.detach().requires_grad_(True)
                        f_i = func(z_i, u_i, t0_i, t1_i)
                        f_x_i = f_i[:, :, :, func.graph_size:]
                        f_x_i = f_x_i[:, :, -1, :]
                        dLdu_i, dLdt0_i, dLdt1_i = torch.autograd.grad(f_x_i, (u_i, t0_i, t1_i),
                                                                               grad_outputs=torch.ones_like(dLdx_i),
                                                                               create_graph=True, allow_unused=True)
                    # Adjusting adjoints with direct gradients
                    adj_z += dLdz_i
                    adj_u += dLdu_i.reshape(bs, n_dim_u)
                    adj_t0[i_t] = adj_t0[i_t] + dLdt0_i.reshape(bs, n_dim_t)
                    adj_t1[i_t] = adj_t1[i_t] + dLdt1_i.reshape(bs, n_dim_t)

                    # Pack augmented variable
                    aug_z = torch.cat((z_i.view(bs, n_dim_z), adj_z,
                                       adj_u, torch.zeros(bs, n_params).to(z), adj_t0[i_t], adj_t1[i_t]), dim=-1)

                    # Solve augmented system backwards
                    aug_ans = ode_solve(aug_z, u, t0_i, t1_i, augmented_dynamics)

                    # Unpack solved backwards augmented system
                    adj_z[:] = aug_ans[:, n_dim_z:
                                          2 * n_dim_z]
                    adj_u[:] = aug_ans[:, 2 * n_dim_z:
                                          2 * n_dim_z + n_dim_u]
                    adj_p[:] += aug_ans[:, 2 * n_dim_z + n_dim_u:
                                           2 * n_dim_z + n_dim_u + n_params]
                    adj_t0[i_t - 1] = aug_ans[:, 2 * n_dim_z + n_dim_u + n_params:
                                                 2 * n_dim_z + n_dim_u + n_params + n_dim_t]
                    adj_t1[i_t - 1] = aug_ans[:, 2 * n_dim_z + n_dim_u + n_params + n_dim_t:
                                                 2 * n_dim_z + n_dim_u + n_params + 2 * n_dim_t]

                    del aug_z, aug_ans

                z_0 = z[0]
                u_0 = u
                t0_0 = t0[0]
                t1_0 = t1[0]
                dLdz_0 = dLdz[0]
                dLdx_0 = dLdz_0.view(bs, *z_shape)[:, :, :, func.graph_size:]
                dLdx_0 = dLdx_0[:, :, -1, :]
                with torch.set_grad_enabled(True):
                    t0_0 = t0_0.detach().requires_grad_(True)
                    t1_0 = t1_0.detach().requires_grad_(True)
                    z_0 = z_0.detach().requires_grad_(True)
                    u_0 = u_0.detach().requires_grad_(True)
                    f_0 = func(z_0, u_0, t0_0, t1_0)
                    f_x_0 = f_0[:, :, :, func.graph_size:]
                    f_x_0 = f_x_0[:, :, -1, :]
                    dLdu_0, dLdt0_0, dLdt1_0 = torch.autograd.grad(f_x_0, (u_0, t0_0, t1_0),
                                                                   grad_outputs=torch.ones_like(dLdx_0),
                                                                   create_graph=True)
                # Adjust adjoints
                adj_z += dLdz_0
                adj_u += dLdu_0.reshape(bs, n_dim_u)
                adj_t0[0] = adj_t0[0] + dLdt0_0.reshape(bs, n_dim_t)
                adj_t1[0] = adj_t1[0] + dLdt1_0.reshape(bs, n_dim_t)
                ## Adjust 0 time adjoint with direct gradients
                # Compute direct gradients

        return adj_z.view(bs, *z_shape), adj_u.view(bs, *u_shape), \
               adj_t0.view(bs, *t_shape), adj_t1.view(bs, *t_shape), adj_p, None

class NeuralODE(nn.Module):
    def __init__(self, func, n_poi):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func
        self.n_poi = n_poi

    def forward(self, data, fuse_embed, return_whole_sequence=False):
        A = data.input_freq.unsqueeze(1).repeat(1, 3, 1, 1)
        x = fuse_embed[data.input_seq].transpose(1, 2)
        u = fuse_embed[self.n_poi + data.uid]

        bs, *A_shape = A.shape
        A = A.reshape(bs, -1)
        x = x.reshape(bs, -1)
        u = u.reshape(bs, -1)

        t0 = data.cur_time
        t1 = data.tar_time
        iter_num = self.func.time_embedding.freq_num

        period = [torch.ceil(self.func.time_embedding.sups[i] - self.func.time_embedding.infs[i]).int() for i in range(iter_num)]

        # max_step = np.trunc(np.log([365*24, 30*24, 7*24, 24, 1])).astype(np.int64) + 2

        # max_step = [0, 3, 3, 3, 3]
        max_step = [4] * iter_num
        # for i in range(1, 5):
        #     max_step[i] = np.random.randint(2, 7)

        def linspace(start, end, i):
            if start > end:
                return torch.linspace(start, end + period[i].item(), max_step[i]).unsqueeze(0) % period[i].item()
            else:
                return torch.linspace(start, end, max_step[i]).unsqueeze(0)

        T = []
        t0_ = t0
        for j in range(0, iter_num):
            inter = [linspace(t0_[i, j], t1[i, j], j) for i in range(bs)]
            inter = torch.cat(inter, dim=0)
            for k in range(max_step[j]):
                t0_[:, j] = inter[:, k]
                T.append(t0_.unsqueeze(0))

        T = torch.cat(T, dim=0)
        timestamp_T = self.func.time_embedding(T)
        z = torch.cat([A, x, u], dim=-1)
        # z = ODEAdjoint.apply(torch.cat([A, x], dim=-1), u, timestamp_T[:-1], timestamp_T[1:], self.func.flatten_parameters(), self.func)
        # t0 = self.func.time_embedding(t0)
        # t1 = self.func.time_embedding(t1)
        # z = z + self.func(z, t0, t1)
        for i in range(np.sum(max_step) - 1):
            z = z + self.func(z, timestamp_T[i], timestamp_T[i + 1])
        if return_whole_sequence:
            return z
        else:
            return z
