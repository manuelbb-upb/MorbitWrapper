function [y1, y2] = matlab_objectives( x_vec )
	y1 = sum( ( x_vec - 1 ).^2 );
	y2 = sum( ( x_vec + 1 ).^2 );
end
