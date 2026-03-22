syms vt vn R alpha omega omega2 vk theta W H vx vy;
A=[sin(theta),cos(theta);cos(theta),-sin(theta)]*[R,cos(alpha);0,sin(alpha)];
B=[vx-omega*W/2;vy+omega*H/2]
simplify(A*[omega2;vk])
