import numpy as np

class EpidemicSimulator:
    def __init__(self, N, L, r0, lam, model_type, w0=1.0, gamma=1.0):
        self.N = N
        self.L = L
        self.r0 = r0
        self.lam = lam
        self.model_type = model_type
        self.w0 = w0
        self.gamma = gamma
        
        self.positions = np.random.rand(N, 2) * L
        self.positions[0] = [L / 2.0, 0.0] # F0 ở giữa đáy
        
        self.is_superspreader = np.random.rand(N) < lam
        #self.is_superspreader[0] = False 
        
        self.states = np.zeros(N, dtype=int)
        self.states[0] = 1 
        
        self.is_percolated = False                    
        self.rf_history = []                          
        self.daily_new_cases = []                     
        self.secondary_infections = np.zeros(N, int)  

    def _distance_periodic(self, p1, p2):
        """Tính khoảng cách dạng hình trụ (Chỉ tuần hoàn trục X)"""
        # Trục X tuần hoàn
        dx = np.abs(p1[:, 0] - p2[0])
        dx = np.minimum(dx, self.L - dx)
        # Trục Y giữ nguyên khoảng cách tuyệt đối
        dy = np.abs(p1[:, 1] - p2[1])
        return np.sqrt(dx**2 + dy**2)

    def run(self):
        while np.any(self.states == 1):
            infected_indices = np.where(self.states == 1)[0]
            susceptible_indices = np.where(self.states == 0)[0]
            
            # ĐO LƯỜNG TIỀN TUYẾN rf: Đo tất cả những ai đã từng nhiễm bệnh (states >= 1)
            infected_ever = np.where(self.states >= 1)[0]
            if len(infected_ever) > 0:
                # Phải dùng công thức tuần hoàn trục X để tính đúng khoảng cách từ F0
                dx = np.abs(self.positions[infected_ever, 0] - self.L/2.0)
                dx = np.minimum(dx, self.L - dx)
                dy = self.positions[infected_ever, 1] - 0.0 # Khoảng cách Y so với F0
                dist_from_origin = np.sqrt(dx**2 + dy**2)
                
                max_rf = np.max(dist_from_origin)
                self.rf_history.append(max_rf)
                
                # Check thẩm thấu (Percolation)
                max_y = np.max(self.positions[infected_ever, 1])
                if max_y >= self.L - self.r0:
                    self.is_percolated = True
            
            newly_infected = []
            new_infections_this_step = 0

            # QUÁ TRÌNH LÂY NHIỄM
            for i_idx in infected_indices:
                if len(susceptible_indices) == 0:
                    break
                
                distances = self._distance_periodic(self.positions[susceptible_indices], self.positions[i_idx])
                probs = np.zeros(len(distances))
                
                if self.is_superspreader[i_idx]:
                    if self.model_type == 'strong':
                        mask = distances <= self.r0
                        probs[mask] = self.w0
                    elif self.model_type == 'hub':
                        rn = np.sqrt(6) * self.r0
                        mask = distances <= rn
                        probs[mask] = self.w0 * (1 - distances[mask] / rn)**2
                else:
                    mask = distances <= self.r0
                    probs[mask] = self.w0 * (1 - distances[mask] / self.r0)**2
                
                infected_mask = np.random.rand(len(distances)) < probs
                successfully_infected = susceptible_indices[infected_mask]
                
                if len(successfully_infected) > 0:
                    newly_infected.extend(successfully_infected)
                    self.secondary_infections[i_idx] += len(successfully_infected)
                    susceptible_indices = np.setdiff1d(susceptible_indices, successfully_infected)

            # CẬP NHẬT TRẠNG THÁI
            if newly_infected:
                self.states[newly_infected] = 1
                new_infections_this_step = len(newly_infected)
            
            self.daily_new_cases.append(new_infections_this_step)
            
            # PHỤC HỒI
            for i_idx in infected_indices:
                if np.random.rand() < self.gamma:
                    self.states[i_idx] = 2

        return {
            "is_percolated": self.is_percolated,
            "rf_history": self.rf_history,
            "daily_new_cases": self.daily_new_cases,
            "secondary_infections": self.secondary_infections
        }